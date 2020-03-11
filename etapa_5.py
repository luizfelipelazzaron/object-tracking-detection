#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3

v = 0.1  # Velocidade linear
w = 0.5  # Velocidade angular

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)

    try:
        while not rospy.is_shutdown():
            vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
            pub.publish(vel)
            rospy.sleep(2*3.14159265359/2)
            vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
            pub.publish(vel)
            rospy.sleep(20.0)
            vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
            pub.publish(vel)
            rospy.sleep(2.0)
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")



# #! /usr/bin/env python
# #! -*- coding:utf-8 -*-

# import rospy
# from geometry_msgs.msg import Twist, Vector3
# from math import pi

# v = 10 #Velocidade linear do robô
# w = (3)/4 #Velocidade Angular do robô

# if __name__ == "__main__":
#     rospy.init_node("roda código da etapa 5")
#     pub = rospy.Publisher("cmd_vel",Twist,queue_size=3)

#     try:
#         while not rospy.is_shutdown():
#             #O robô Avança para frente
#             vel = Twist(Vector3(v,0,0),Vector3(0,0,0))
#             pub.publish(vel)
#             rospy.sleep(4)
#             #O robô para
#             vel = Twist(Vector3(0,0,0),Vector3(0,0,0))
#             pub.publish(vel)
#             rospy.sleep(4)
#             # O robô rotaciona 90 graus
#             vel = Twist(Vector3(0,0,0),Vector3(0,0,w))
#             pub.publish(vel)
#             rospy.sleep(2)
#     except rospy.ROSInterruptException:
#         print("Ocorreu uma exceção com o rospy")

