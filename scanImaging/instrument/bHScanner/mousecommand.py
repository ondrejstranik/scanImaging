import serial
import time

class CommWindows200Computer:
    def __init__(self, port='COM4', baudrate=9600):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        if not self.ser.is_open:
            raise Exception(f"Could not open serial port {port}")
        self.ser.flushInput()
#        self.ser.reset_input_buffer()
        self.ser.flushOutput()
        self.sleep_time = 0.1
        self.scanner_state=0
        self.coord_once=[105,55]
        self.coord_xyrepeat=[30,55]
        self.coord_zstack_up=[103,719]
        self.coord_zstack_down=[103,743]
        self.coord_zstack_uup=[103,696]
        self.coord_zstack_ddown=[103,766]
    
    def move_mouse(self, x, y):
        command = f'M,{x},{y}'
        self.send_command(command)

    def mouse_button(self, button):
        command = f'B,{button}'
        self.send_command(command)

    def key_press(self, key_code):
        command = f'K,{key_code}'
        #0x50 is virtual key code for 'P'
        self.send_command(command)

    def key_press(self, key_code):
        command = f'K,{key_code}'
        self.send_command(command)

    def shortcut_key_press(self, key_code):
        # strg+alt+key
        command = f'S,{key_code}'
        self.send_command(command)

    def start_fluoview(self):
        #0x50 is virtual key code for 'P'
        self.shortcut_key_press(0x50)
        # wait until open
        time.sleep(5)
        self.window_command_max()

    def window_command_max(self):
        command = 'W,1'
        self.send_command(command)
    
    def window_command_close(self):
        command = 'W,2'
        self.send_command(command)
   
    def move_mouse_press_release(self, x,y):
        self.move_mouse(x, y)
        self.mouse_button(1)
        self.mouse_button(0)

    def stop_scanner(self):
        if self.scanner_state==1:
            self.move_mouse_press_release(self.coord_xyrepeat[0], self.coord_xyrepeat[1])
            self.scanner_state=0

    def start_repeat_scan(self):
        if self.scanner_state==0:
            self.move_mouse_press_release(self.coord_xyrepeat[0], self.coord_xyrepeat[1])
            self.scanner_state=1

    def start_once_scan(self):
        self.move_mouse_press_release(self.coord_once[0], self.coord_once[1])

    def send_command(self, command):
        self.ser.write(command.encode() + b'\n')
        time.sleep(self.sleep_time)

    def scanner_zstack_up(self,uup=False):
        if uup:
            self.move_mouse_press_release(self.coord_zstack_uup[0], self.coord_zstack_uup[1])
        else:
            self.move_mouse_press_release(self.coord_zstack_up[0], self.coord_zstack_up[1])

    def scanner_zstack_down(self,ddown=False):
        if ddown:
            self.move_mouse_press_release(self.coord_zstack_ddown[0], self.coord_zstack_ddown[1])
        else:
            self.move_mouse_press_release(self.coord_zstack_down[0], self.coord_zstack_down[1])


    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    with CommWindows200Computer(port='COM4') as comm:
        print("Sending mouse commands...")
        comm.scanner_zstack_down()
        time.sleep(1)
        comm.scanner_zstack_down(True)
        time.sleep(5)
        comm.scanner_zstack_up(True)
        time.sleep(1)
        comm.scanner_zstack_up()
#        comm.start_fluoview()
 #       comm.start_repeat_scan()
 #       time.sleep(3)
 #       comm.stop_scanner()
 #       time.sleep(1)
 #       comm.start_once_scan()
 #       time.sleep(1)
        print("Commands sent successfully.")

if __name__ == "__main__":
    main()