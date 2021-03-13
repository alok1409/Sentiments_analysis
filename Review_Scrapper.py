import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
import sqlite3 as sql

urls = []
product_urls = []
list_of_reviews = []
driver = webdriver.Firefox(executable_path="C:/Users/HP/Downloads/geckodriver")

def scrap_reviews():
    for i in range(1, 251):
        urls.append(f"https://www.etsy.com/in-en/c/jewelry/earrings/ear-jackets-and-climbers?ref=pagination&explicit=1&page={i}")
    
    for url in urls:
        driver.get(url)
        sleep(5)
        for i in range(1, 65):
            product = driver.find_element_by_xpath(f'//*[@id="content"]/div/div[1]/div/div[3]/div[2]/div[2]/div[1]/div/div/ul/li[{i}]/div/a')
            product_urls.append(product.get_attribute('href'))
    
    for product_url in product_urls[1:]:
        try:
            driver.get(product_url)
            sleep(5)
            html = driver.page_source
            soup = BeautifulSoup(html,'html')
            for i in range(4):
                try:
                    list_of_reviews.append(soup.select(f'#review-preview-toggle-{i}')[0].getText().strip())
                except:
                    continue
            while(True):
                try:
                    next_button = driver.find_element_by_xpath('//*[@id="reviews"]/div[2]/nav/ul/li[position() = last()]/a[contains(@href, "https")]')
                    if next_button != None:
                        next_button.click()
                        sleep(5)
                        html = driver.page_source
                        soup = BeautifulSoup(html,'html')
                        for i in range(4):
                            try: 
                                list_of_reviews.append(soup.select(f'#review-preview-toggle-{i}')[0].getText().strip())
                            except:
                                continue
                except Exception as e:
                    print('finish : ', e)
                    break
        except:
            continue
            
        scrapped_reviews = pd.DataFrame(list_of_reviews, index = None, columns = ['reviews'])         
        scrapped_reviews.to_csv('scrapped_reviews.csv')

def store_in_database():
    df = pd.read_csv('scrapped_reviews.csv')
    conn = sql.connect('Review_db.db')
    df.to_sql('Review_db', conn)       

def main():
    global urls
    global product_urls
    global list_of_reviews
    global driver
    print("Starting Review Scrapping...")
    scrap_reviews()
    print("Review Scrapped and now Storing in Database...")
    store_in_database()
    print("All the Reviews are Store in Database")
    urls = None
    product_urls = None
    list_of_reviews = None
    driver = None
if __name__=='__main__':
    main()



