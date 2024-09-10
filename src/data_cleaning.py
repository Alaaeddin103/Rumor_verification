import json
import re

class DatasetCleaner:
    def __init__(self, issue_texts=None):
       
        if issue_texts is None:
           
            self.issue_texts = [
                self.normalize_text("ISSUE: couldn't translate"),  
                self.normalize_text("ISSUE : could n't translate"), 
                self.normalize_text("ISSUE : couldn't translate")    
            ]
        else:
            # Normalize issue texts
            self.issue_texts = [self.normalize_text(issue_text) for issue_text in issue_texts]

    def remove_irrelevant_rumors(self, data):
        
        cleaned_data = []
        for rumor in data:
            timeline = rumor.get("timeline", [])
            # Normalize and filter the timeline, removing invalid entries
            cleaned_timeline = [entry for entry in timeline if self.normalize_text(entry[2]) not in self.issue_texts]

            # Check if the cleaned timeline has valid entries
            if cleaned_timeline:
                # If there are valid timeline entries, keep the rumor and update its timeline
                rumor["timeline"] = cleaned_timeline
                cleaned_data.append(rumor)
            # If all entries were invalid, the rumor is excluded from cleaned_data automatically

        return cleaned_data

    def normalize_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip())  # Normalize multiple spaces
        text = re.sub(r'\s*:\s*', ': ', text)  # Normalize spaces around colons ("ISSUE : could" to "ISSUE: could")
        return text

    def save_cleaned_data(self, cleaned_data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)