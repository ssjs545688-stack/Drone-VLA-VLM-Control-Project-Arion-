#!/usr/bin/env python3
import math,json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String,Float32MultiArray

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('rgb_topic','/camera/image_raw')
        self.declare_parameter('depth_topic','/camera/depth_image')
        self.declare_parameter('camera_info_topic','/camera/camera_info')
        self.declare_parameter('target_color','red')
        self.declare_parameter('min_area',400)
        self.declare_parameter('sync_slop',1.0)
        self.declare_parameter('hsv_h_min',0)
        self.declare_parameter('hsv_s_min',100)
        self.declare_parameter('hsv_v_min',100)
        self.declare_parameter('hsv_h_max',10)
        self.declare_parameter('hsv_s_max',255)
        self.declare_parameter('hsv_v_max',255)

        self.rgb_topic=self.get_parameter('rgb_topic').value
        self.depth_topic=self.get_parameter('depth_topic').value
        self.camera_info_topic=self.get_parameter('camera_info_topic').value
        self.target_color=self.get_parameter('target_color').value
        self.min_area=self.get_parameter('min_area').value

        self.fx=self.fy=None
        self.camera_frame_id='camera_optical_frame'

        self.latest_target_info={
            'detected':False,
            'x':0.0,'y':0.0,'z':0.0,
            'distance':0.0,
            'relative_yaw_deg':0.0,
            'relative_yaw_rad':0.0
        }

        self.latest_display_image=None
        self.latest_cv_depth=None
        self.latest_depth_stamp=None
        self.received_frames=0
        self.camera_window_created=False

        self.create_timer(1.0,self.log_target_info)
        self.create_timer(0.05,self.gui_timer_callback)

        self.target_json_pub=self.create_publisher(
            String,'/detected_object/info',10
        )
        self.target_array_pub=self.create_publisher(
            Float32MultiArray,'/detected_object/data',10
        )
        self.target_point_pub=self.create_publisher(
            PointStamped,'/detected_object/point',10
        )
        self.debug_image_pub=self.create_publisher(
            Image,'/detected_object/debug_image',10
        )

        qos=QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos
        )
        self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_image_callback,
            qos
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_image_callback,
            qos
        )

        self.get_logger().info(
            f'[ObjectDetector] Started. RGB: {self.rgb_topic}, Depth: {self.depth_topic}'
        )

    def imgmsg_to_numpy(self,msg):
        if msg.encoding in ('32FC1','32FC3'):
            dtype=np.float32
        elif msg.encoding in ('16UC1','16UC3','mono16'):
            dtype=np.uint16
        else:
            dtype=np.uint8

        if msg.encoding in ('mono8','mono16') or msg.encoding.endswith('C1'):
            channels=1
        elif msg.encoding.endswith('C3'):
            channels=3
        elif msg.encoding.endswith('C4'):
            channels=4
        else:
            channels=3

        arr=np.frombuffer(msg.data,dtype=dtype)

        if channels==1:
            return arr.reshape(msg.height,msg.width)

        return arr.reshape(msg.height,msg.width,channels)

    def numpy_to_imgmsg(self,img,header,encoding='bgr8'):
        msg=Image()
        msg.header=header
        msg.height=img.shape[0]
        msg.width=img.shape[1]
        msg.encoding=encoding
        msg.is_bigendian=False
        msg.step=img.strides[0]
        msg.data=img.tobytes()
        return msg

    def gui_timer_callback(self):
        if self.latest_target_info['detected'] and self.latest_display_image is not None:
            if not self.camera_window_created:
                cv2.namedWindow('Gazebo Camera',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Gazebo Camera',960,720)
                self.camera_window_created=True

            cv2.imshow('Gazebo Camera',self.latest_display_image)
            cv2.waitKey(1)

        elif self.camera_window_created:
            try:
                cv2.destroyWindow('Gazebo Camera')
            except Exception:
                pass

            self.camera_window_created=False

    def log_target_info(self):
        t=self.latest_target_info

        if t['detected']:
            self.get_logger().info(
                f"[Object] x={t['x']:.3f} m, y={t['y']:.3f} m, z={t['z']:.3f} m, "
                f"distance={t['distance']:.3f} m, "
                f"relative_yaw_deg={t['relative_yaw_deg']:+.2f} deg, "
                f"relative_yaw_rad={t['relative_yaw_rad']:+.4f} rad"
            )
        else:
            self.get_logger().info(
                f'[Object] No target detected (Frames received: {self.received_frames})'
            )

    def camera_info_callback(self,msg):
        if self.fx is None:
            self.fx=msg.k[0]
            self.fy=msg.k[4]

            if msg.header.frame_id:
                self.camera_frame_id=msg.header.frame_id

            self.get_logger().info(
                f'[ObjectDetector] CameraInfo: fx={self.fx:.2f}, fy={self.fy:.2f}'
            )

    def depth_image_callback(self,depth_msg):
        try:
            cv_d=self.imgmsg_to_numpy(depth_msg)

            if depth_msg.encoding=='16UC1':
                cv_d=cv_d.astype(np.float32)/1000.0
            else:
                cv_d=cv_d.astype(np.float32)

            self.latest_cv_depth=cv_d
            self.latest_depth_stamp=depth_msg.header.stamp

        except Exception as e:
            self.get_logger().error(
                f'Depth image conversion error: {e}'
            )

    def get_color_mask(self,hsv_img):
        color=self.target_color.lower()

        if color=='red':
            mask1=cv2.inRange(
                hsv_img,
                np.array([0,100,70]),
                np.array([10,255,255])
            )
            mask2=cv2.inRange(
                hsv_img,
                np.array([170,100,70]),
                np.array([180,255,255])
            )
            mask=cv2.bitwise_or(mask1,mask2)

        elif color=='green':
            mask=cv2.inRange(
                hsv_img,
                np.array([35,80,70]),
                np.array([85,255,255])
            )

        elif color=='blue':
            mask=cv2.inRange(
                hsv_img,
                np.array([100,100,70]),
                np.array([130,255,255])
            )

        elif color=='yellow':
            mask=cv2.inRange(
                hsv_img,
                np.array([20,100,100]),
                np.array([35,255,255])
            )

        elif color=='orange':
            mask=cv2.inRange(
                hsv_img,
                np.array([10,100,100]),
                np.array([25,255,255])
            )

        elif color=='custom':
            hmin=self.get_parameter('hsv_h_min').value
            smin=self.get_parameter('hsv_s_min').value
            vmin=self.get_parameter('hsv_v_min').value
            hmax=self.get_parameter('hsv_h_max').value
            smax=self.get_parameter('hsv_s_max').value
            vmax=self.get_parameter('hsv_v_max').value

            mask=cv2.inRange(
                hsv_img,
                np.array([hmin,smin,vmin]),
                np.array([hmax,smax,vmax])
            )

        else:
            mask1=cv2.inRange(
                hsv_img,
                np.array([0,100,70]),
                np.array([10,255,255])
            )
            mask2=cv2.inRange(
                hsv_img,
                np.array([170,100,70]),
                np.array([180,255,255])
            )
            mask=cv2.bitwise_or(mask1,mask2)

        kernel=cv2.getStructuringElement(
            cv2.MORPH_RECT,(5,5)
        )
        mask=cv2.morphologyEx(
            mask,cv2.MORPH_OPEN,kernel
        )
        mask=cv2.morphologyEx(
            mask,cv2.MORPH_CLOSE,kernel
        )

        return mask

    def rgb_image_callback(self,rgb_msg):
        try:
            cv_rgb=self.imgmsg_to_numpy(rgb_msg)

            if rgb_msg.encoding=='rgb8':
                cv_rgb=cv2.cvtColor(
                    cv_rgb,cv2.COLOR_RGB2BGR
                )
            elif rgb_msg.encoding!='bgr8':
                if rgb_msg.encoding=='rgba8':
                    cv_rgb=cv2.cvtColor(
                        cv_rgb,cv2.COLOR_RGBA2BGR
                    )
                elif rgb_msg.encoding=='bgra8':
                    cv_rgb=cv2.cvtColor(
                        cv_rgb,cv2.COLOR_BGRA2BGR
                    )
                elif rgb_msg.encoding=='mono8':
                    cv_rgb=cv2.cvtColor(
                        cv_rgb,cv2.COLOR_GRAY2BGR
                    )
                else:
                    self.get_logger().warn(
                        f'Unsupported RGB encoding: {rgb_msg.encoding}'
                    )
                    return

        except Exception as e:
            self.get_logger().error(
                f'RGB image conversion error: {e}'
            )
            return

        h,w,_=cv_rgb.shape

        # 실제 RGB 영상의 중앙을 카메라 중심으로 사용
        cx=w/2.0
        cy=h/2.0

        # 초점거리는 CameraInfo 사용
        fx=self.fx if self.fx is not None else (
            (w/2.0)/math.tan(math.radians(35.0))
        )
        fy=self.fy if self.fy is not None else (
            (h/2.0)/math.tan(math.radians(35.0))
        )

        hsv=cv2.cvtColor(cv_rgb,cv2.COLOR_BGR2HSV)
        mask=self.get_color_mask(hsv)

        contours,_=cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        target_info={
            'detected':False,
            'x':0.0,
            'y':0.0,
            'z':0.0,
            'distance':0.0,
            'relative_yaw_deg':0.0,
            'relative_yaw_rad':0.0,
            'pixel_u':0,
            'pixel_v':0
        }

        debug_img=cv_rgb.copy()

        if contours:
            largest_contour=max(
                contours,
                key=cv2.contourArea
            )
            area=cv2.contourArea(largest_contour)

            if area>=self.min_area:
                bx,by,bw,bh=cv2.boundingRect(
                    largest_contour
                )
                M=cv2.moments(largest_contour)

                if M['m00']>0:
                    u=int(M['m10']/M['m00'])
                    v=int(M['m01']/M['m00'])
                else:
                    u=int(bx+bw/2)
                    v=int(by+bh/2)

                u=max(0,min(w-1,u))
                v=max(0,min(h-1,v))

                Z=0.0
                cv_depth=self.latest_cv_depth

                if cv_depth is not None:
                    dh,dw=cv_depth.shape[:2]

                    du=int(u*dw/w)
                    dv=int(v*dh/h)

                    du_min=max(0,du-2)
                    du_max=min(dw,du+3)
                    dv_min=max(0,dv-2)
                    dv_max=min(dh,dv+3)

                    depth_patch=cv_depth[
                        dv_min:dv_max,
                        du_min:du_max
                    ]

                    valid_depths=depth_patch[
                        np.isfinite(depth_patch)&
                        (depth_patch>0.05)
                    ]

                    if len(valid_depths)>0:
                        Z=float(np.median(valid_depths))
                    elif dv<dh and du<dw:
                        center_depth=cv_depth[dv,du]

                        if np.isfinite(center_depth) and center_depth>0:
                            Z=float(center_depth)

                if Z>0.05:
                    X=(u-cx)*Z/fx
                    Y=(v-cy)*Z/fy

                    distance=math.sqrt(
                        X**2+Y**2+Z**2
                    )

                    # 카메라 영상 중심을 yaw=0 기준으로 사용
                    relative_yaw_rad=math.atan2(X,Z)
                    relative_yaw_deg=math.degrees(
                        relative_yaw_rad
                    )

                    target_info={
                        'detected':True,
                        'x':round(X,3),
                        'y':round(Y,3),
                        'z':round(Z,3),
                        'distance':round(distance,3),
                        'relative_yaw_deg':round(
                            relative_yaw_deg,2
                        ),
                        'relative_yaw_rad':round(
                            relative_yaw_rad,4
                        ),
                        'pixel_u':u,
                        'pixel_v':v
                    }

                    cv2.rectangle(
                        debug_img,
                        (bx,by),
                        (bx+bw,by+bh),
                        (0,255,0),
                        3
                    )

                    cv2.circle(
                        debug_img,
                        (u,v),
                        7,
                        (0,0,255),
                        -1
                    )

                    cv2.putText(
                        debug_img,
                        f'Dist: {distance:.2f}m',
                        (bx,max(35,by-45)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0,255,255),
                        3
                    )

                    cv2.putText(
                        debug_img,
                        f'Yaw: {relative_yaw_deg:+.1f} deg',
                        (bx,max(70,by-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0,255,255),
                        3
                    )

                    cv2.putText(
                        debug_img,
                        f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}',
                        (bx,by+bh+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,255,0),
                        2
                    )

                else:
                    target_info={
                        'detected':True,
                        'x':0.0,
                        'y':0.0,
                        'z':0.0,
                        'distance':0.0,
                        'relative_yaw_deg':0.0,
                        'relative_yaw_rad':0.0,
                        'pixel_u':u,
                        'pixel_v':v
                    }

                    cv2.rectangle(
                        debug_img,
                        (bx,by),
                        (bx+bw,by+bh),
                        (0,255,0),
                        3
                    )

                    cv2.circle(
                        debug_img,
                        (u,v),
                        7,
                        (0,0,255),
                        -1
                    )

                    cv2.putText(
                        debug_img,
                        'Target (No Depth)',
                        (bx,max(35,by-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0,255,255),
                        3
                    )

        self.latest_target_info=target_info

        # 카메라 영상 정중앙 = yaw 0도 기준
        cv2.drawMarker(
            debug_img,
            (int(cx),int(cy)),
            (255,0,0),
            markerType=cv2.MARKER_CROSS,
            markerSize=35,
            thickness=2
        )

        json_msg=String()
        json_msg.data=json.dumps(target_info)
        self.target_json_pub.publish(json_msg)

        array_msg=Float32MultiArray()
        array_msg.data=[
            float(target_info['x']),
            float(target_info['y']),
            float(target_info['z']),
            float(target_info['distance']),
            float(target_info['relative_yaw_deg']),
            float(target_info['relative_yaw_rad']),
            1.0 if target_info['detected'] else 0.0
        ]
        self.target_array_pub.publish(array_msg)

        if target_info['detected']:
            pt_msg=PointStamped()
            pt_msg.header.stamp=rgb_msg.header.stamp
            pt_msg.header.frame_id=self.camera_frame_id
            pt_msg.point.x=float(target_info['x'])
            pt_msg.point.y=float(target_info['y'])
            pt_msg.point.z=float(target_info['z'])
            self.target_point_pub.publish(pt_msg)

        try:
            debug_img_msg=self.numpy_to_imgmsg(
                debug_img,
                rgb_msg.header,
                'bgr8'
            )
            self.debug_image_pub.publish(
                debug_img_msg
            )
        except Exception as e:
            self.get_logger().error(
                f'Debug image conversion error: {e}'
            )

        self.latest_display_image=debug_img
        self.received_frames+=1

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node=ObjectDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            '[ObjectDetector] Shutting down...'
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()