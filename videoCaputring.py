import cv2

vid = cv2.VideoCapture(0)

output_folder = "/Users/sid/Desktop/Project/Videos/test.mp4"
codec = cv2.VideoWriter_fourcc(*'MJPG')
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(output_folder, codec, 24.0 , (width,height))
frame_count = 0
max_frame = 400

while frame_count < max_frame:
    try:
        success,frame = vid.read()
        if not success :
            print('unable to access the frame ')
            break

        video_writer.write(frame)
        cv2.imshow('Video', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    except Exception as e:
        print('Some error occured' , e)
        break

vid.release()
video_writer.release()
cv2.destroyAllWindows()