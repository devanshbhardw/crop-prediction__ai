from app import app, db, User, Prediction, Contact
from datetime import datetime

def export_database_to_txt():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = "database_export.txt"
    
    with app.app_context(), open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"CropPredict Database Export\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write("="*50 + "\n\n")

        # Export Users
        f.write("USERS\n")
        f.write("-"*50 + "\n")
        users = User.query.all()
        if users:
            for user in users:
                f.write(f"User ID: {user.id}\n")
                f.write(f"Username: {user.username}\n")
                f.write(f"Email: {user.email}\n")
                f.write(f"Full Name: {user.full_name or 'Not set'}\n")
                f.write(f"Organization: {user.organization or 'Not set'}\n")
                f.write(f"Job Title: {user.job_title or 'Not set'}\n")
                f.write(f"Number of Predictions: {len(user.predictions)}\n")
                f.write("-"*30 + "\n")
        else:
            f.write("No users found in database.\n")
        f.write("\n")

        # Export Predictions
        f.write("PREDICTIONS\n")
        f.write("-"*50 + "\n")
        predictions = Prediction.query.all()
        if predictions:
            for pred in predictions:
                f.write(f"Prediction ID: {pred.id}\n")
                f.write(f"Date: {pred.date}\n")
                f.write(f"Crop: {pred.crop}\n")
                f.write(f"Yield Value: {pred.yield_value:.2f} tons/hectare\n")
                f.write("Environmental Parameters:\n")
                f.write(f"  Temperature: {pred.temperature}°C\n")
                f.write(f"  Humidity: {pred.humidity}%\n")
                f.write(f"  Rainfall: {pred.rainfall}mm\n")
                f.write("Soil Parameters:\n")
                f.write(f"  Nitrogen: {pred.nitrogen} kg/ha\n")
                f.write(f"  Phosphorus: {pred.phosphorus} kg/ha\n")
                f.write(f"  Potassium: {pred.potassium} kg/ha\n")
                f.write(f"  pH: {pred.ph}\n")
                f.write(f"Made by: {pred.author.username}\n")
                f.write("-"*30 + "\n")
        else:
            f.write("No predictions found in database.\n")
        f.write("\n")

        # Export Contact Submissions
        f.write("CONTACT SUBMISSIONS\n")
        f.write("-"*50 + "\n")
        contacts = Contact.query.all()
        if contacts:
            for contact in contacts:
                f.write(f"Contact ID: {contact.id}\n")
                f.write(f"Name: {contact.name}\n")
                f.write(f"Email: {contact.email}\n")
                f.write(f"Phone: {contact.phone}\n")
                f.write(f"Date: {contact.date}\n")
                f.write(f"Message:\n{contact.message}\n")
                f.write("-"*30 + "\n")
        else:
            f.write("No contact submissions found in database.\n")

    print(f"Database exported successfully to {output_file}")
    return output_file

if __name__ == "__main__":
    export_database_to_txt()