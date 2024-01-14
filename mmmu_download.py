from datasets import load_dataset
from tqdm import tqdm

cate = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
temp = {
    "Art": "art_and_design",
    "Design": "art_and_design",
    "Music": "art_and_design",
    "Art_Theory": "art_and_design",
    "Accounting": "business",
    "Economics": "business",
    "Finance": "business",
    "Manage": "business",
    "Marketing": "business",
    "Biology": "science",
    "Chemistry": "science",
    "Geography": "science",
    "Math": "science",
    "Physics": "science",
    "Basic_Medical_Science": "health_and_medicine",
    "Clinical_Medicine": "health_and_medicine",
    "Diagnostics_and_Laboratory_Medicine": "health_and_medicine",
    "Pharmacy": "health_and_medicine",
    "Public_Health": "health_and_medicine",
    "History": "humanities_and_social_sci",
    "Literature": "humanities_and_social_sci",
    "Psychology": "humanities_and_social_sci",
    "Sociology": "humanities_and_social_sci",
    "Agriculture": "tech_and_engineering",
    "Architecture_and_Engineering": "tech_and_engineering",
    "Computer_Science": "tech_and_engineering",
    "Electronics": "tech_and_engineering",
    "Energy_and_Power": "tech_and_engineering",
    "Materials": "tech_and_engineering",
    "Mechanical_Engineering": "tech_and_engineering"

}
ids = 0
save_dir = "./LLaVA/mmmu"
pattern = r"\['(.*?)'\]"
for c in tqdm(cate):
    dataset = load_dataset("MMMU/MMMU", c)
    splits = ['dev', 'test', 'validation']
    for s in splits:
        images = dataset[s]['image_1']
        for img in images:
            path = f"{save_dir}/{ids}.png"
            img.save(path)
            ids += 1
