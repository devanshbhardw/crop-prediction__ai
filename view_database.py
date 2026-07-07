from app import app, db, User, Prediction, Contact

def view_database():
    with app.app_context():
        print("\n=== USERS ===")
        users = User.query.all()
        for user in users:
            print(f"\nUser ID: {user.id}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
            print(f"Full Name: {user.full_name}")
            print(f"Organization: {user.organization}")
            print(f"Job Title: {user.job_title}")
            print(f"Number of Predictions: {len(user.predictions)}")

        print("\n=== PREDICTIONS ===")
        predictions = Prediction.query.all()
        for pred in predictions:
            print(f"\nPrediction ID: {pred.id}")
            print(f"Date: {pred.date}")
            print(f"Crop: {pred.crop}")
            print(f"Yield Value: {pred.yield_value}")
            print(f"Temperature: {pred.temperature}°C")
            print(f"Humidity: {pred.humidity}%")
            print(f"Rainfall: {pred.rainfall}mm")
            print(f"Soil Parameters:")
            print(f"  N: {pred.nitrogen} kg/ha")
            print(f"  P: {pred.phosphorus} kg/ha")
            print(f"  K: {pred.potassium} kg/ha")
            print(f"  pH: {pred.ph}")
            print(f"User: {pred.author.username}")

        print("\n=== CONTACT SUBMISSIONS ===")
        contacts = Contact.query.all()
        for contact in contacts:
            print(f"\nContact ID: {contact.id}")
            print(f"Name: {contact.name}")
            print(f"Email: {contact.email}")
            print(f"Phone: {contact.phone}")
            print(f"Date: {contact.date}")
            print(f"Message: {contact.message}")

if __name__ == "__main__":
    view_database()