import requests
from bs4 import BeautifulSoup
import csv

# URL of the UIUC Course Explorer
BASE_URL = "https://courses.illinois.edu"
COURSES_ENDPOINT = f"{BASE_URL}/schedule/DEFAULT/DEFAULT"

# Simulate scraping data from the website
def scrape_course_catalog():
    print("Scraping course catalog from UIUC Course Explorer...")

    # Simulate a GET request to fetch the HTML content (placeholder URL)
    response = requests.get(COURSES_ENDPOINT)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Simulate parsing the data (adjust selectors to match actual site structure)
    courses = []
    course_rows = soup.find_all("div", class_="course-row")  # Example selector
    for row in course_rows:
        course_code = row.find("span", class_="course-code").text.strip()
        course_title = row.find("span", class_="course-title").text.strip()
        instructor = row.find("span", class_="instructor").text.strip()
        credits = row.find("span", class_="credits").text.strip()
        description = row.find("div", class_="description").text.strip()
        schedule = row.find("span", class_="schedule").text.strip()
        location = row.find("span", class_="location").text.strip()

        courses.append({
            "Course Code": course_code,
            "Course Title": course_title,
            "Instructor": instructor,
            "Credits": credits,
            "Description": description,
            "Schedule": schedule,
            "Location": location,
        })

    return courses

# File name for the output CSV
output_file = "course_catalog.csv"

# Writing the data to a CSV file
def write_to_csv(course_data):
    if not course_data:
        print("No course data to write.")
        return

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=course_data[0].keys())
        writer.writeheader()
        writer.writerows(course_data)

    print(f"Course catalog has been written to {output_file}.")

# Main function to scrape and save the course catalog
def main():
    course_data = scrape_course_catalog()
    write_to_csv(course_data)

if __name__ == "__main__":
    main()
