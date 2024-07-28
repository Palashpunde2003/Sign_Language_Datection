import cv2
import os

DATA_DIR = '/Users/palashpunde/Documents/Sing_Language_Detection'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class_types = 37
batch_size = 100

class_names = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',
              6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',
              12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
              18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',
              24:'Y',25:'Z',26:'_'}

cap = cv2.VideoCapture(0)
for j in range(len(class_names)):
    if not os.path.exists(os.path.join(DATA_DIR, class_names[j])):
        os.makedirs(os.path.join(DATA_DIR, class_names[j]))

    print('Collecting data for class {}'.format(class_names[j]))

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Press "S" to Start Collecion !', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow("Data Collection",frame)
        if cv2.waitKey(25) & 0xFF == ord('s'):
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0
    while counter < batch_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, class_names[j], '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()