import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO
import cv2

# COCO class IDs and names
coco_classes = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat",
    "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
    "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
    "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
    "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork",
    "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog",
    "Pizza", "Donut", "Cake", "Chair", "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV",
    "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator",
    "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
]

# Function to find class index (case-insensitive)
def find_class_index(class_name):
    lower_classes = [cls.lower() for cls in coco_classes]
    try:
        return lower_classes.index(class_name.lower())
    except ValueError:
        return -1  # Return -1 if not found

# Initialize the recognizer and TTS engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Function to convert text to speech
def SpeakText(command):
    engine.say(command)
    engine.runAndWait()

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Load YOLOv8 model

# Path to the image
image_path = "./input_image.jpg"
original_image = cv2.imread(image_path)


# Function to display detections for the spoken class
def display_detections(target_class_id):
    image = original_image.copy()
    results = model(image_path)  # Perform object detection

    # Filter results for the specified class ID
    filtered_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == target_class_id:  # Filter by class ID
                filtered_boxes.append(box)

    # Draw filtered boxes on the image
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(
            image,
            f"{coco_classes[target_class_id]} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Show the image with updated detections
    cv2.imshow("Real-Time Detection", image)

# Main loop for voice input and real-time detection
while True:
    try:
        # Use the microphone as source for input
        with sr.Microphone() as source2:
            print("Listening...")
            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2)
            MyText = r.recognize_google(audio2).lower()

            print(f"Did you say: {MyText}")
            SpeakText(f"You said: {MyText}")

            # Find class ID from voice input
            target_class_id = find_class_index(MyText)
            if target_class_id == -1:
                print(f"Class '{MyText}' not found in the COCO dataset.")
                SpeakText("Object not found in the dataset. Try again.")
                continue

            print(f"Class '{MyText}' corresponds to ID: {target_class_id}")

            # Display detections for the spoken class
            display_detections(target_class_id)

            # Wait for a key press to continue listening
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")

# Cleanup
cv2.destroyAllWindows()
