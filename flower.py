import cv2
import tensorflow as tf

# Load the trained TensorFlow model
model = tf.keras.models.load_model('flower_classifier.h5')

# Initialize the webcam using OpenCV
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    frame2 = frame

    # Preprocess the frame
    frame = cv2.resize(frame, (180, 180)) / 255.0
    frame = frame.reshape(1, 180, 180, 3)
    
    # Predict the class of the frame using the model
    prediction = model.predict(frame)
    
    # Get the class with the highest probability
    print(prediction)
    predicted_class = tf.argmax(prediction[0]).numpy()
    
    # Print the predicted class on the frame
    
    
    # Display the frame
    frame = frame.reshape(180, 180, 3)
    cv2.putText(frame, "Class: " + str(predicted_class), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()