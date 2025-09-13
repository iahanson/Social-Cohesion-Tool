"""
Alert System for Local Authorities
SMS/Email alert system for social tension monitoring
"""

import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AlertSystem:
    """Alert system for notifying local authorities of social tension risks"""
    
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_username = os.getenv('EMAIL_USERNAME')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Alert thresholds
        self.thresholds = {
            'critical_risk': 0.8,
            'high_risk': 0.6,
            'medium_risk': 0.4,
            'anomaly_threshold': -0.5
        }
        
        # Alert history
        self.alert_history = []
        
    def create_alert_message(self, alert_data: Dict) -> str:
        """
        Create formatted alert message
        
        Args:
            alert_data: Dictionary with alert information
            
        Returns:
            Formatted alert message string
        """
        alert_type = alert_data.get('type', 'UNKNOWN')
        msoa_code = alert_data.get('msoa_code', 'N/A')
        local_authority = alert_data.get('local_authority', 'N/A')
        priority = alert_data.get('priority', 'MEDIUM')
        
        message = f"""
🚨 SOCIAL COHESION ALERT 🚨

Alert Type: {alert_type}
Priority: {priority}
MSOA Code: {msoa_code}
Local Authority: {local_authority}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if 'risk_score' in alert_data:
            message += f"Risk Score: {alert_data['risk_score']:.2f}\n"
        
        if 'anomaly_score' in alert_data:
            message += f"Anomaly Score: {alert_data['anomaly_score']:.2f}\n"
        
        message += f"\nAlert Message:\n{alert_data.get('message', 'No additional details')}\n"
        
        if 'recommended_actions' in alert_data:
            message += "\nRecommended Actions:\n"
            for i, action in enumerate(alert_data['recommended_actions'], 1):
                message += f"{i}. {action}\n"
        
        message += f"\n---\nThis is an automated alert from the Social Cohesion Monitoring System."
        
        return message
    
    def send_email_alert(self, recipients: List[str], alert_data: Dict, 
                        subject_prefix: str = "Social Cohesion Alert") -> bool:
        """
        Send email alert to recipients
        
        Args:
            recipients: List of email addresses
            alert_data: Dictionary with alert information
            subject_prefix: Subject line prefix
            
        Returns:
            True if successful, False otherwise
        """
        if not self.email_username or not self.email_password:
            print("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"{subject_prefix} - {alert_data.get('priority', 'MEDIUM')} Priority"
            
            # Create message body
            message_body = self.create_alert_message(alert_data)
            msg.attach(MIMEText(message_body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_username, recipients, text)
            server.quit()
            
            # Log successful send
            self._log_alert('email', recipients, alert_data, 'success')
            return True
            
        except Exception as e:
            print(f"Error sending email alert: {e}")
            self._log_alert('email', recipients, alert_data, 'failed', str(e))
            return False
    
    def send_sms_alert(self, phone_numbers: List[str], alert_data: Dict) -> bool:
        """
        Send SMS alert to phone numbers
        
        Args:
            phone_numbers: List of phone numbers
            alert_data: Dictionary with alert information
            
        Returns:
            True if successful, False otherwise
        """
        if not self.twilio_account_sid or not self.twilio_auth_token:
            print("Twilio credentials not configured")
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            # Create SMS message (truncated for SMS)
            sms_message = f"🚨 SOCIAL COHESION ALERT 🚨\n"
            sms_message += f"Priority: {alert_data.get('priority', 'MEDIUM')}\n"
            sms_message += f"MSOA: {alert_data.get('msoa_code', 'N/A')}\n"
            sms_message += f"LA: {alert_data.get('local_authority', 'N/A')}\n"
            sms_message += f"Risk Score: {alert_data.get('risk_score', 'N/A')}\n"
            sms_message += f"Time: {datetime.now().strftime('%H:%M')}\n"
            sms_message += f"Details: {alert_data.get('message', 'Check email for details')}"
            
            # Send SMS to each number
            for phone_number in phone_numbers:
                message = client.messages.create(
                    body=sms_message,
                    from_=self.twilio_phone_number,
                    to=phone_number
                )
                print(f"SMS sent to {phone_number}: {message.sid}")
            
            # Log successful send
            self._log_alert('sms', phone_numbers, alert_data, 'success')
            return True
            
        except Exception as e:
            print(f"Error sending SMS alert: {e}")
            self._log_alert('sms', phone_numbers, alert_data, 'failed', str(e))
            return False
    
    def _log_alert(self, alert_type: str, recipients: List[str], alert_data: Dict, 
                   status: str, error_message: str = None):
        """
        Log alert attempt
        
        Args:
            alert_type: Type of alert ('email' or 'sms')
            recipients: List of recipients
            alert_data: Alert data
            status: 'success' or 'failed'
            error_message: Error message if failed
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'recipients': recipients,
            'alert_data': alert_data,
            'status': status,
            'error_message': error_message
        }
        
        self.alert_history.append(log_entry)
        
        # Save to file
        self._save_alert_log()
    
    def _save_alert_log(self):
        """Save alert log to file"""
        try:
            log_file = 'data/alert_log.json'
            os.makedirs('data', exist_ok=True)
            
            with open(log_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            print(f"Error saving alert log: {e}")
    
    def load_alert_log(self) -> List[Dict]:
        """Load alert log from file"""
        try:
            log_file = 'data/alert_log.json'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    self.alert_history = json.load(f)
            return self.alert_history
        except Exception as e:
            print(f"Error loading alert log: {e}")
            return []
    
    def check_alert_frequency(self, msoa_code: str, alert_type: str, 
                            time_window_hours: int = 24) -> bool:
        """
        Check if alerts have been sent recently for this MSOA
        
        Args:
            msoa_code: MSOA code to check
            alert_type: Type of alert
            time_window_hours: Time window in hours
            
        Returns:
            True if alerts can be sent, False if too frequent
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if (alert['alert_data'].get('msoa_code') == msoa_code and
                alert['alert_type'] == alert_type and
                datetime.fromisoformat(alert['timestamp']) > cutoff_time and
                alert['status'] == 'success')
        ]
        
        # Allow alerts if less than 3 in the time window
        return len(recent_alerts) < 3
    
    def process_alerts(self, alerts: List[Dict], 
                      email_recipients: List[str] = None,
                      sms_recipients: List[str] = None) -> Dict:
        """
        Process a list of alerts and send notifications
        
        Args:
            alerts: List of alert dictionaries
            email_recipients: List of email addresses
            sms_recipients: List of phone numbers
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'total_alerts': len(alerts),
            'email_sent': 0,
            'sms_sent': 0,
            'email_failed': 0,
            'sms_failed': 0,
            'skipped_frequency': 0,
            'processed_alerts': []
        }
        
        for alert in alerts:
            msoa_code = alert.get('msoa_code', 'unknown')
            priority = alert.get('priority', 'MEDIUM')
            
            processed_alert = {
                'msoa_code': msoa_code,
                'priority': priority,
                'email_sent': False,
                'sms_sent': False,
                'skipped': False
            }
            
            # Check alert frequency
            if not self.check_alert_frequency(msoa_code, 'email'):
                processed_alert['skipped'] = True
                results['skipped_frequency'] += 1
                results['processed_alerts'].append(processed_alert)
                continue
            
            # Send email alerts
            if email_recipients and priority in ['HIGH', 'CRITICAL']:
                if self.send_email_alert(email_recipients, alert):
                    processed_alert['email_sent'] = True
                    results['email_sent'] += 1
                else:
                    results['email_failed'] += 1
            
            # Send SMS alerts for critical alerts
            if sms_recipients and priority == 'CRITICAL':
                if self.send_sms_alert(sms_recipients, alert):
                    processed_alert['sms_sent'] = True
                    results['sms_sent'] += 1
                else:
                    results['sms_failed'] += 1
            
            results['processed_alerts'].append(processed_alert)
        
        return results
    
    def create_alert_summary_report(self, time_window_hours: int = 24) -> Dict:
        """
        Create summary report of recent alerts
        
        Args:
            time_window_hours: Time window for report
            
        Returns:
            Dictionary with alert summary
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        if not recent_alerts:
            return {
                'time_window_hours': time_window_hours,
                'total_alerts': 0,
                'summary': 'No alerts in the specified time window'
            }
        
        # Analyze alerts
        alert_types = {}
        priorities = {}
        local_authorities = {}
        msoa_codes = {}
        
        for alert in recent_alerts:
            alert_data = alert['alert_data']
            
            # Count by type
            alert_type = alert_data.get('type', 'unknown')
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            # Count by priority
            priority = alert_data.get('priority', 'unknown')
            priorities[priority] = priorities.get(priority, 0) + 1
            
            # Count by local authority
            la = alert_data.get('local_authority', 'unknown')
            local_authorities[la] = local_authorities.get(la, 0) + 1
            
            # Count by MSOA
            msoa = alert_data.get('msoa_code', 'unknown')
            msoa_codes[msoa] = msoa_codes.get(msoa, 0) + 1
        
        return {
            'time_window_hours': time_window_hours,
            'total_alerts': len(recent_alerts),
            'alert_types': alert_types,
            'priorities': priorities,
            'local_authorities': local_authorities,
            'most_alerted_msoas': dict(sorted(msoa_codes.items(), key=lambda x: x[1], reverse=True)[:5]),
            'success_rate': len([a for a in recent_alerts if a['status'] == 'success']) / len(recent_alerts),
            'email_alerts': len([a for a in recent_alerts if a['alert_type'] == 'email']),
            'sms_alerts': len([a for a in recent_alerts if a['alert_type'] == 'sms'])
        }
    
    def configure_alert_settings(self, settings: Dict):
        """
        Configure alert system settings
        
        Args:
            settings: Dictionary with configuration settings
        """
        if 'thresholds' in settings:
            self.thresholds.update(settings['thresholds'])
        
        if 'smtp_server' in settings:
            self.smtp_server = settings['smtp_server']
        
        if 'smtp_port' in settings:
            self.smtp_port = settings['smtp_port']
        
        print("Alert system settings updated")
    
    def test_alert_system(self, test_email: str = None, test_phone: str = None) -> Dict:
        """
        Test the alert system with sample data
        
        Args:
            test_email: Test email address
            test_phone: Test phone number
            
        Returns:
            Dictionary with test results
        """
        test_alert = {
            'type': 'SYSTEM_TEST',
            'msoa_code': 'E02000001',
            'local_authority': 'Test Authority',
            'priority': 'MEDIUM',
            'message': 'This is a test alert from the Social Cohesion Monitoring System',
            'recommended_actions': [
                'Verify alert system is working',
                'Check email/SMS delivery',
                'Review alert configuration'
            ]
        }
        
        results = {
            'test_alert_created': True,
            'email_test': False,
            'sms_test': False,
            'email_error': None,
            'sms_error': None
        }
        
        # Test email
        if test_email:
            try:
                results['email_test'] = self.send_email_alert([test_email], test_alert, "TEST ALERT")
            except Exception as e:
                results['email_error'] = str(e)
        
        # Test SMS
        if test_phone:
            try:
                results['sms_test'] = self.send_sms_alert([test_phone], test_alert)
            except Exception as e:
                results['sms_error'] = str(e)
        
        return results
