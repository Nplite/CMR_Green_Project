import smtplib
import sys
from MetalTheft.constant import *
from MetalTheft.logger import logging
from MetalTheft.exception import MetalTheptException
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders



class EmailSender:
    try:
        def __init__(self, sender_email=SENDER_EMAIL, receiver_email=RECEIVER_EMAIL, smtp_login_email_pass=SMTP_LOGIN_EMAIL_PASS):
            self.sender_email = sender_email
            self.receiver_email = receiver_email
            self.smtp_login_email_pass = smtp_login_email_pass

        def send_alert_email(self, attachment_path=None, video_url=None, camera_id=None):
            try:
                subject = "Regarding the Metal Object Theft"
                # Update the message to include the video URL if provided
                message = f"""Hi,
We have observed an incident on This camera IP number: {camera_id}, here Someone is throwing a metal object over the wall.

Here is the video link for the incident: {video_url if video_url else "Video not available"}

Best Regards,
AI Security"""

                msg = MIMEMultipart()
                msg['From'] = self.sender_email
                msg['To'] = self.receiver_email
                msg['Subject'] = subject

                msg.attach(MIMEText(message, 'plain'))

                # Attach the video file if provided
                if attachment_path:
                    with open(attachment_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename={attachment_path}')
                        msg.attach(part)

                # Connect to the server and send the email
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(self.sender_email, self.smtp_login_email_pass)
                text = msg.as_string()
                server.sendmail(self.sender_email, self.receiver_email, text)
                server.quit()

                logging.info(f"Alert  Email has been sent to {camera_id}")
                return f"Email has been sent to: {self.receiver_email}"
            
            except Exception as e:
                raise MetalTheptException(e, sys) from e


        def daily_report_email(self, attachment_path=None):
            try:
                subject = "Regarding the Metal Object Theft"
                message = """These is the daily report of the mails."""
                msg = MIMEMultipart()
                msg['From'] = self.sender_email
                msg['To'] = self.receiver_email
                msg['Subject'] = subject

                msg.attach(MIMEText(message, 'plain'))

                if attachment_path:
                    attachment = open(attachment_path, "rb")
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={attachment_path}')
                    msg.attach(part)
                    attachment.close()

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(self.sender_email, self.smtp_login_email_pass)
                text = msg.as_string()
                server.sendmail(self.sender_email, self.receiver_email, text)
                server.quit()
                logging.info(f"Daily Report Email has been sent to: {self.receiver_email}")

                return f"Daily Report Email has been sent to: {self.receiver_email}"
            
            except Exception as e:
                raise MetalTheptException(e, sys) from e




        def monthly_report_email(self, attachment_path=None):
            try:
                subject = "Regarding the Metal Object Theft"
                message = """These is the daily report of the mails."""
                msg = MIMEMultipart()
                msg['From'] = self.sender_email
                msg['To'] = self.receiver_email
                msg['Subject'] = subject

                msg.attach(MIMEText(message, 'plain'))

                if attachment_path:
                    attachment = open(attachment_path, "rb")
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={attachment_path}')
                    msg.attach(part)
                    attachment.close()

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(self.sender_email, self.smtp_login_email_pass)
                text = msg.as_string()
                server.sendmail(self.sender_email, self.receiver_email, text)
                server.quit()
                logging.info(f"Monthly Report Email has been sent to: {self.receiver_email}")

                return f"Monthly Report Email has been sent to: {self.receiver_email}"
            except Exception as e:
                raise MetalTheptException(e, sys) from e



    except Exception as e:
        raise MetalTheptException(e, sys) from e



