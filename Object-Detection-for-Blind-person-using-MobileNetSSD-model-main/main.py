import cv2
import numpy as np
import random
import pyttsx3  # Text-to-speech
import speech_recognition as sr
import threading
import queue
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import openrouteservice  #OpenRouteService API
import requests  # For weather API
import datetime



engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Queue for managing speech commandsq
speech_queue = queue.Queue()

# OpenRouteService API client
ORS_API_KEY = "5b3ce3597851110001cf624850cec98c6de14b318f9811cf83e36e53"
ors_client = openrouteservice.Client(key=ORS_API_KEY)

# OpenWeatherMap API Key
OPENWEATHERMAP_API_KEY = "e46ed23a821f20778b44a9ca036468c6"

# Random Colors for Bounding Boxe
R = random.randint(0, 255)
G = random.randint(0, 255)
B = random.randint(0, 255)

# Video Capture
video = cv2.VideoCapture(0)

# Object Classes for Detection
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

color = [(R, G, B) for _ in CLASSES]
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Detection Control Flag
detection_active = False
# Keep Track of Detected Objects
detected_objects = set()


current_location = (11.3410, 77.7172)  

# Geolocator for GPS
geolocator = Nominatim(user_agent="geoassistant")

# Function to speak detected objects in a separate thread
def speak_thread():
    while True:
        text = speech_queue.get()
        if text == "exit":
            break
        engine.say(text)
        engine.runAndWait()

def announce_date_time():
    now = datetime.datetime.now()
    day = now.strftime("%A")
    time = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")
    speech_queue.put(f"Today is {day}, the date is {date}, and the current time is {time}.")

# Start the speech thread
speech_thread = threading.Thread(target=speak_thread)
speech_thread.daemon = True
speech_thread.start()

# Determine Object Direction~exit
def get_direction(mid_x, frame_width):
    if mid_x < frame_width / 3:
        return "left"
    elif mid_x > 2 * frame_width / 3:
        return "right"
    else:
        return "center"

# Announce Current Location
def announce_current_location():
    try:
        location = geolocator.reverse(current_location)
        if location:
            address = location.address
            speech_queue.put(f"You are currently at {address}.")
        else:
            speech_queue.put("Unable to determine your current location.")
    except Exception as e:
        speech_queue.put("Error retrieving your current location.")

# Fetch Weather Data
def get_weather_data(location=None):
    try:
        if location:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        else:
            lat, lon = current_location
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            city = data["name"]
            temperature = data["main"]["temp"]
            weather_description = data["weather"][0]["description"]
            speech_queue.put(f"The current weather in {city} is {weather_description} with a temperature of {temperature} degrees Celsius.")
        else:
            speech_queue.put("Unable to fetch weather data. Please try again later.")
    except Exception as e:
        speech_queue.put("Error retrieving weather information.")
        print(f"Weather API Error: {e}")
def get_navigation_to(destination_address):
    try:
        destination = geolocator.geocode(destination_address)
        if destination:
            destination_coords = (destination.latitude, destination.longitude)
            distance = geodesic(current_location, destination_coords).kilometers
            speech_queue.put(f"The destination is {distance:.2f} kilometers away.")

            # Fetch route from OpenRouteService API
            coords = [current_location, destination_coords]
            route = ors_client.directions(coordinates=coords, profile='foot-walking', format='geojson')
            
            # Get turn-by-turn instructions
            steps = route['features'][0]['properties']['segments'][0]['steps']
            for step in steps:
                instruction = step['instruction']
                distance = step['distance']
                speech_queue.put(f"{instruction}. Distance: {distance:.1f} meters.")
        else:
            speech_queue.put("Destination not found.")
    except Exception as e:
        speech_queue.put("Error retrieving navigation data.")

# Voice Command Listener
def listen_for_commands():
    global detection_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    speech_queue.put("Voice command system activated. Say 'Start detection', 'Stop detection', 'Where am I?', 'Weather in [city]', or 'Navigate to [destination]'.")

    while True:
        with mic as source:
            print("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source)

            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                if "start detection" in command:
                    detection_active = True
                    speech_queue.put("Detection started.")
                elif "stop detection" in command:
                    detection_active = False
                    speech_queue.put("Detection stopped.")
                elif "where am i" in command:
                    announce_current_location()
                elif "navigate to" in command:
                    destination_address = command.replace("navigate to", "").strip()
                    speech_queue.put(f"Navigating to {destination_address}.")
                    get_navigation_to(destination_address)
                elif "weather in" in command:
                    city = command.replace("weather in", "").strip()
                    speech_queue.put(f"Fetching weather information for {city}.")
                    get_weather_data(city)
                elif "current weather" in command:
                    speech_queue.put("Fetching current weather information.")
                    get_weather_data()
                elif "date and time" in command:
                    announce_date_time()
                elif "exit" in command or "quit" in command:
                    speech_queue.put("Exiting the program.")
                    break
            except sr.WaitTimeoutError:
                print("Listening timed out, no command detected.")
            except sr.UnknownValueError:
                print("Could not understand the command.")
            except sr.RequestError:
                print("Speech recognition service is unavailable.")

# Main Loop for Object Detection
command_thread = threading.Thread(target=listen_for_commands)
command_thread.daemon = True
command_thread.start()

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    (h, w) = frame.shape[:2]

    if detection_active:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = CLASSES[idx]

                mid_x = (startX + endX) / 2
                direction = get_direction(mid_x, w)

                if label not in detected_objects:
                    speech_queue.put(f"{label} detected on your {direction} with {int(confidence * 100)} percent confidence.")
                    detected_objects.add(label)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color[idx], 2)
                label_with_confidence = f"{label}: {int(confidence * 100)}%"
                cv2.putText(frame, label_with_confidence, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()