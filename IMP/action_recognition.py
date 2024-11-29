

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

class MoViNetClassifier:
    def __init__(self, model_id='a0', num_classes=10, resolution=224, num_frames=8, weights_path="MetalTheft/Movinet_Model.keras"):
        self.num_frames = num_frames
        self.resolution = resolution
        self.frame_buffer = deque(maxlen=num_frames)
        
        # Load the model
        backbone = movinet.Movinet(model_id=model_id)
        backbone.trainable = False
        self.model = self.build_classifier(backbone, num_classes)
        self.model.build([1, num_frames, resolution, resolution, 3])
        self.model.load_weights(weights_path)
        
        # Define class names (adjust this based on your dataset)
        self.class_names = ["CrikectBowling", "JavelinThrow", "ThrowDiscuss"]

    def build_classifier(self, backbone, num_classes):
        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=num_classes
        )
        return model

    @tf.function
    def predict(self, input_frames):
        return self.model(input_frames, training=False)

    def format_frames(self, frame):
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, self.resolution, self.resolution)
        return frame

    def process_frame(self, frame):
        processed_frame = self.format_frames(frame)
        self.frame_buffer.append(processed_frame)

        if len(self.frame_buffer) == self.num_frames:
            model_input = tf.stack(list(self.frame_buffer))
            model_input = tf.expand_dims(model_input, axis=0)
            prediction = self.predict(model_input)
            predicted_class = self.class_names[np.argmax(prediction[0])]
            confidence = tf.nn.softmax(prediction[0])[np.argmax(prediction[0])].numpy()
            return predicted_class, confidence
        return None, None

# Example of using the MoViNetClassifier in another video capture code
def main():
    classifier = MoViNetClassifier()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    prev_time = 0

    with tf.device('/GPU:0'):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time

            # Process frame and get predictions
            predicted_class, confidence = classifier.process_frame(frame)
            
            # Display results if available
            if predicted_class:
                cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('MoViNet_test', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
