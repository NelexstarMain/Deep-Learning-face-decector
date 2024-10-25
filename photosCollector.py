import cv2
import os

FOLDER_PATH = "faces/"
ID: str = ""

def create_person(id: str) -> None:
    global ID
    ID = id
    path = os.path.join(FOLDER_PATH, id)
    os.makedirs(path, exist_ok=True) 
    print(f"Directory for ID: {id} created.")

create_person(input("whats your name? "))

def create_person_photos() -> None:
    distance: str = "ok"
    frame_count = 0
    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()
        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
            faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

            
            for (x, y, w, h) in faces_rect:
                if x:
                    face_img = frame[y:y + h, x:x + w]
                    if face_img.shape[:2] < (100, 100):
                        distance = "Too small face"
                        
                    elif face_img.shape[:2] > (250, 250):
                        distance = "Too big face"
                        
                    else:
                        distance = "Face detected"
                        filename = f"{FOLDER_PATH}/{ID}/frame_{frame_count}.jpg"
                        print(filename)
                        cv2.imwrite(filename, face_img)
                    cv2.putText(frame, f"{(frame_count/1500)*100:.4
                                          
                                          
                                          
                                          f}%", (10, 20),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    
                    cv2.putText(frame, distance, (x, y - 20),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 3)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow(f"Creating Person '{ID}'", frame)
            frame_count += 1
        
            if frame_count >= 1500:
                break
        else:
            print("Failed to read frame. Skipping this iteration.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_person_photos()