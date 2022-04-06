import smtplib
from email import encoders
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage

def send_email(Address,Mess):

    email = "camera.surveillance.ai@gmail.com"
    password = "wsknesnxguxmnbnd"
    address = Address
    msg = Mess
    

    client = smtplib.SMTP("smtp.gmail.com",587)
    client.starttls()
    client.login(email,password)
    client.sendmail(email,address,msg)
    print("Đã gửi cảnh báo đến:"+ address)
    client.quit()

