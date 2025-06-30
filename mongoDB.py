from pymongo import MongoClient

client = MongoClient("mongodb+srv://hidheshp2:KMAHjLb0Uk82mTrL@cluster0.flfg6iv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["college_db"]
collection = db["full_details"]

# Insert one document
#collection.insert_one({"name": "Alice", "age": 25})

# Insert many documents
collection.insert_many([
    {
  "college": {
    "name": "ABC Institute of Technology",
    "affiliation": "National Board of Higher Education (NBHE)",
    "location": "123 Tech Park, Silicon Valley, CA",
    "accreditation": "Accredited by NAAC with Grade 'A'",
    "facilities": [
      "Modern classrooms",
      "Computer labs",
      "Library",
      "Hostel",
      "Sports Complex",
      "Wi-Fi",
      "Campus"
    ]
  },
  "courses": [
    {
      "name": "Python for Beginners",
      "department": "Computer Science",
      "duration": "3 Months",
      "eligibility": "Basic Computer Knowledge",
      "syllabus": [
        "Introduction to Python",
        "Data Types",
        "Control Flow",
        "Functions",
        "OOP in Python",
        "File Handling"
      ],
      "admission": {
        "deadline": "2025-03-04",
        "documents": [
          "ID Proof",
          "Passport Size Photo"
        ],
        "process_link": "http://127.0.0.1:8000/admin/"
      },
      "fees": {
        "total": 20000.0,
        "installment": 1000,
        "scholarship": "Merit-Based Scholarship"
      }
    },
    {
      "name": "Django REST Framework",
      "department": "Information Technology",
      "duration": "3 Months",
      "eligibility": "Basic Computer Knowledge",
      "syllabus": [
        "Django Basics",
        "REST API",
        "Authentication",
        "Serializers",
        "Viewsets",
        "Deployment"
      ],
      "admission": {
        "deadline": "2025-03-29",
        "documents": [
          "ID Proof",
          "Passport Size Photo"
        ],
        "process_link": "http://127.0.0.1:8000/admin/"
      }
    },
    {
      "name": "Machine Learning Basics",
      "department": "Artificial Intelligence",
      "duration": "4 Months",
      "eligibility": "Python & Math Fundamentals",
      "syllabus": [
        "Regression",
        "Classification",
        "Clustering",
        "Neural Networks",
        "Model Evaluation"
      ],
      "admission": {
        "deadline": "2025-03-29",
        "documents": [
          "ID Proof",
          "Passport Size Photo"
        ],
        "process_link": "http://127.0.0.1:8000/admin/"
      }
    },
    {
      "name": "React for Beginners",
      "department": "Electronics & Communication",
      "duration": "4 Months",
      "eligibility": "Basic HTML, CSS, JavaScript",
      "syllabus": [
        "React Components",
        "Props & State",
        "Hooks",
        "Routing",
        "API Integration",
        "Redux"
      ],
      "admission": {
        "deadline": "2025-03-29",
        "documents": [
          "ID Proof",
          "Passport Size Photo"
        ],
        "process_link": "http://127.0.0.1:8000/admin/"
      }
    }
  ],
  "departments": [
    {
      "name": "Computer Science",
      "hod": "Dr. Rajesh Kumar",
      "email": "cs_office@college.edu",
      "location": "Block A"
    },
    {
      "name": "Information Technology",
      "hod": "Dr. Priya Sharma",
      "email": "it_office@college.edu",
      "location": "Block B"
    },
    {
      "name": "Artificial Intelligence",
      "hod": "Dr. Anil Mehta",
      "email": "ai_office@college.edu",
      "location": "Block C"
    },
    {
      "name": "Electronics & Communication",
      "hod": "Dr. Neha Verma",
      "email": "ece_office@college.edu",
      "location": "Block D"
    }
  ],
  "scholarship": {
    "name": "Merit-Based Scholarship",
    "criteria": "Minimum 90% in previous academic year",
    "application_link": "https://chatgpt.com/c/67d7f260-9714-8002-b2e2-69fc89a6780f",
    "amount": 5000.0
  }
}
])
