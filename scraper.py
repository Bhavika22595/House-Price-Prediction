import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException

def convert_to_number(price_txt):
    if 'Call for Price' in price_txt:
        return None
    elif 'Cr' in price_txt:
        number = float(price_txt.replace('₹', '').replace('Cr', '').strip()) * 10_000_000
    elif 'Lac' in price_txt:
        number = float(price_txt.replace('₹', '').replace('Lac', '').strip()) * 100_000
    else:
        number = float(price_txt.replace('₹', '').strip())
    return number

def convert_area_to_sqft(area):
    if 'sqyrd' in area:
        area_value = float(area.replace('sqyrd', '').strip())
        area_in_sqft = area_value * 9
        return area_in_sqft
    else:
        return float(area.replace('sqft', '').strip())

def close_popup(driver):
    try:
        popup = driver.find_element(By.XPATH, "//div[contains(@class, 'npsSidePopCon')]")
        close_button = popup.find_element(By.XPATH, ".//button[contains(@class, 'close')]")
        close_button.click()
        time.sleep(1)  # Allow time for popup to close
    except NoSuchElementException:
        pass

def extract_info_and_scroll(driver, writer, place, area_sqft):
    try:
        price_txt = driver.find_element(By.XPATH, "//div[@class='mb-ldp_more-dtl_list--value']").text
        price_part = price_txt.split('|')[0].strip()
        price = convert_to_number(price_part)
    except NoSuchElementException:
        price = None

    NULL = None

    try:
        bedrooms_text = driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_summary--item ico-beds']").text
        bedrooms_str = ''.join(char for char in bedrooms_text if char.isdigit())
        bedrooms = int(bedrooms_str) if bedrooms_str else NULL
    except NoSuchElementException:
        bedrooms = NULL

    try:
        bathrooms_text = driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_summary--item ico-baths']").text
        bathrooms_str = ''.join(char for char in bathrooms_text if char.isdigit())
        bathrooms = int(bathrooms_str) if bathrooms_str else NULL
    except NoSuchElementException:
        bathrooms = NULL

    try:
        balconies_text = driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_summary--item ico-balconies']").text
        balconies_str = ''.join(char for char in balconies_text if char.isdigit())
        balconies = int(balconies_str) if balconies_str else NULL
    except NoSuchElementException:
        balconies = NULL

    try:
        car_text = driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_summary--item ico-covered-parking']").text
        car_str = ''.join(char for char in car_text if char.isdigit())
        car_parking = int(car_str) if car_str else NULL
    except NoSuchElementException:
        car_parking = NULL

    try:
        face = driver.find_element(By.XPATH, "//div[contains(text(), 'Facing')]/following-sibling::* ")
        facing = face.text
    except NoSuchElementException:
        facing = NULL

    try:
        view_all_link = driver.find_element(By.CLASS_NAME, "mb-ldp__more-dtl--viewall")
        view_all_link.click()
        time.sleep(2)
    except NoSuchElementException:
        pass

    try:
        main_road_element = driver.find_element(By.XPATH, "//div[@class='mb-ldp_more-dtl_list--value' and contains(text(), 'Main Road')]")
        near_main_road = "Yes"
    except NoSuchElementException:
        near_main_road = "No"

    try:
        if driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_list--value' and contains(text(), 'Unfurnished')]"):
            furnished_status = "Unfurnished"
        elif driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_list--value' and contains(text(), 'Semi-Furnished')]"):
            furnished_status = "Semi-Furnished"
        else:
            furnished_status = "Furnished"
    except (NoSuchElementException, StaleElementReferenceException):
        furnished_status = "Furnished"

    try:
        transaction = driver.find_element(By.XPATH, "//div[@class='mb-ldp_dtlsbody_list--value' and contains(text(), 'Resale')]")
        transaction_type = "Resale"
    except NoSuchElementException:
        transaction_type = "New Property"

    writer.writerow([price, bedrooms, bathrooms, balconies, area_sqft, place, furnished_status, near_main_road, transaction_type, car_parking, facing])

def main():
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    csv_file_path = os.path.join(downloads_path, 'magicbricks_properties.csv')

    driver = webdriver.Chrome()
    url = "https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=%3E5,1,2,3,4&proptype=Residential-House,Villa&cityName=Hyderabad"
    driver.get(url)
    time.sleep(5)

    processed_addresses = set()

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Price', 'Bedrooms', 'Bathrooms', 'Balconies', 'Carpet Area (sqft)', 'Place', 'Furnished_status', 'Main Road', 'Transaction Type', 'Car Parking', 'Facing'])

        while True:
            try:
                addresses = driver.find_elements(By.XPATH, "//h2[@class='mb-srp__card--title']")
            except NoSuchElementException:
                break  # Exit loop if no addresses are found

            for address in addresses:
                try:
                    title = address.text
                    if title in processed_addresses:
                        continue

                    place = title.split("in", 1)[1].strip() if "in" in title else "Unknown"
                    parent_div = address.find_element(By.XPATH, "./ancestor::div[contains(@class, 'mb-srp__card')]")
                    summary_label = parent_div.find_element(By.XPATH, ".//div[@class='mb-srp_card_summary--label']")
                    if summary_label.text == 'CARPET AREA':
                        area = parent_div.find_element(By.XPATH, ".//div[@class='mb-srp_card_summary--value']").text
                        area_sqft = convert_area_to_sqft(area)
                        attempts = 0
                        clicked = False
                        while attempts < 3 and not clicked:
                            try:
                                address.click()
                                clicked = True
                            except ElementClickInterceptedException:
                                close_popup(driver)
                                attempts += 1

                        if not clicked:
                            continue

                        time.sleep(5)
                        driver.switch_to.window(driver.window_handles[1])
                        target_element = driver.find_element(By.XPATH, "//div[@class='mb-ldp_section_title--text1']")

                        extract_info_and_scroll(driver, writer, place, area_sqft)
                        processed_addresses.add(title)
                        time.sleep(5)
                        driver.close()

                        driver.switch_to.window(driver.window_handles[0])

                except (StaleElementReferenceException, NoSuchElementException) as e:
                    print(f"Exception occurred: {e}")
                    continue

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)

            try:
                new_addresses = driver.find_elements(By.XPATH, "//h2[@class='mb-srp__card--title']")
                if len(new_addresses) == len(addresses):
                    break
            except NoSuchElementException:
                break

    driver.quit()
    print(f"Data saved to '{csv_file_path}'")

if _name_ == "_main_":
    main()
