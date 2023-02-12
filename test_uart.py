import serial
import serial.tools.list_ports

def robot_control_uart(direction, depth):
    """  
    This function return the data frame of UART protocol as the control command for the robot
    The data frame is 8-bit wide, with the first 2 bits as the direction command and the last 
    6 bits as the depth information

    -----------------------------------------------------------------------------------------

    Direction encoding: Stop - 0, Right - 1, Left - 2
    Depth encoding: 0 -> 0.5m: 000000
                    0.55m    : 000001
                    0.60m    : 000010\n
                    ...
                    3.65m    : 111111                      
    """
    
    # Direction bits will take 2 bits of the data frame
    direction_bits = bin(direction)[2:]
    
    # Depth bits will be the last 6 bits
    min_threshold = 0.5
    max_threshold = 3.65
    depth = min(depth, max_threshold)
    depth_unit = 0.05
    depth_PU = depth/depth_unit

    depth_bits = "{0:06b}".format(int(depth_PU-min_threshold/depth_unit))

    # Create the data frame
    data_frame = direction_bits + depth_bits
    return data_frame

ports = serial.tools.list_ports.comports()

ser_1 = serial.Serial(port="COM1", baudrate=9600, parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, timeout=1)
                    
ser_2 = serial.Serial(port="COM2", baudrate=9600, parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, timeout=1)

test = robot_control_uart(2, 1.32).encode()
print(test)
i=0

while (i<8):
    ser_1.write(robot_control_uart(2, 1.32).encode())
    receive = ser_2.read()
    print(receive.decode())
    i+=1

