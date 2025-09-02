# import cv2
# import torch

# model = torch.hub.load('D:/Vchanel/Box_detection_web/yolov5', 'custom',
#                        path='D:/Vchanel/Box_detection_web/backend/best_demo_allbox.pt', source='local')
# model.eval()

# cap = cv2.VideoCapture("C:/vchanel/demo/JL_CROP_3.mp4")
# CONF_THRESHOLD = 0.8

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     detections = results.pandas().xyxy[0]

#     for _, row in detections.iterrows():
#         if row['confidence'] >= CONF_THRESHOLD:
#             xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
#             label = f"{row['name']} {row['confidence']:.2f}"
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("Live Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


def send_count_email(count: int, to_email: str):
    logger.info(f"Preparing to send email for count: {count}")
    
    sender_email = "abinayabi55@gmail.com"
    sender_password = "vepmrbuzypehztdh"  # Your Gmail app password
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = "Video Object Count Notification"

    body = f"The total count of objects crossing the line is: {count}"
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(1)  # Enable SMTP debug info
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message.as_string())
        server.quit()
        logger.info("Email sent successfully!")
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP auth error: {e}")
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
