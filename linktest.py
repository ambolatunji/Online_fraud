import requests
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import os

def check_url_details(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=5)
        remark = "Good" if response.status_code == 200 else "Bad"
        return {
            'URL': url,
            'Status': response.status_code,
            'Valid': response.status_code == 200,
            'Remark': remark,
            'Content-Type': response.headers.get('Content-Type', ''),
            'Content-Length': len(response.content),
            'Error': ''
        }
    except requests.exceptions.RequestException as e:
        return {
            'URL': url,
            'Status': "Error",
            'Valid': False,
            'Remark': "Bad",
            'Content-Type': '',
            'Content-Length': 0,
            'Error': str(e)
        }

def bulk_check(urls, max_workers=5, output_csv=True):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(check_url_details, urls))
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"url_check_results_{timestamp}.csv"
    
    if output_csv:
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['URL', 'Status', 'Valid', 'Remark', 'Content-Type', 'Content-Length', 'Error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\nResults saved to: {os.path.abspath(filename)}")
    
    return results
# Example Usage
urls_to_check = [
    "https://www.researchgate.net/publication/388009174_Influence_Tactics_of_Leaders_The_Role_of_Ray_Kroc%27s_Transformational_Leadership_in_Shaping_McDonald%27s_Organizational_Culture_and_Global_Success",
"https://www.linkedin.com/pulse/ive-hired-hundreds-leaders-what-i-look-chris-kempczinski-cukfc",
"https://www.forbes.com/sites/noahbarsky/2022/01/25/mcdonalds-digital-transformation-special-sauce-is-curiosity/",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/CK-leadership-july.html",
"https://www.linkedin.com/pulse/your-leadership-glass-half-full-chris-kempczinski-qixic",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/technology.html",
"https://www.ericgoldman.name/en/2008/observation-of-leadership-and-organizational-behavior-at-mcdonalds/",
"https://www.ukessays.com/essays/management/analysis-of-mcdonalds-leadership-style-management-essay.php",
"https://www.linkedin.com/pulse/leadership-fast-food-hr-insights/",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://corporate.mcdonalds.com/corpmcd/our-people/franchise.html",
"https://trainingmag.com/mcdonalds-gen-z-training/",
"https://corporate.mcdonalds.com/corpmcd/technology/blog",
"https://www.linkedin.com/pulse/importance-transparency-leadership-tips-hr-leaders-patel",
"https://corporate.mcdonalds.com/corpmcd/our-people/education-programs.html",
"https://www.diversityinc.com/mcdonalds-unveils-new-diversity-equity-and-inclusion-initiatives/",
"https://hbr.org/2021/05/4-ways-managers-can-build-psychological-safety",
"https://www.cnbc.com/2023/01/26/mcdonalds-bonuses-tied-to-dei-progress.html",
"https://www.fastcompany.com/90809050/what-gen-z-really-wants-at-work",
"https://corporate.mcdonalds.com/corpmcd/investor-relations/annual-reports.html",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/safety.html",
"https://ivypanda.com/essays/mcdonalds-company-employee-engagement/",
"https://ivypanda.com/essays/mcdonalds-company-employee-engagement/",
"https://www.vaia.com/en-us/explanations/business-studies/business-case-studies/mcdonalds-organisational-structure/",
"https://www.forbes.com/sites/blakemorgan/2022/03/21/why-customer-experience-is-the-new-brand-currency/",
"https://www.qsrmagazine.com/workforce/why-better-leadership-keeps-employees",
"https://corporate.mcdonalds.com/corpmcd/our-people/franchise.html",
"https://www.linkedin.com/pulse/leadership-fast-food-hr-insights/",
"https://www.glassdoor.com/employers/blog/mcdonalds-employee-reviews/",
"https://corporate.mcdonalds.com/corpmcd/investor-relations/annual-reports.html",
"https://www.forbes.com/sites/forbestechcouncil/2023/03/13/tech-rollouts-and-employee-fatigue-in-qsr/",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/safety.html",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/global-standards.html",
"https://www.cnbc.com/2024/02/10/fast-food-turnover-costs.html",
"https://hbr.org/2023/07/franchise-leadership-stress",
"https://www.forbes.com/sites/forbestechcouncil/2024/01/ai-adoption-challenges",
"https://trainingmag.com/mcdonalds-training-gaps/",
"https://www.businessinsider.com/mcdonalds-wage-impacts-2024",
"https://www.bbc.com/news/business-50214217",
"https://www.glassdoor.com/Reviews/McDonald-s-Manager-Hours",
"https://www.osha.gov/mcdonalds-compliance",
"https://www.mckinsey.com/industries/retail/our-insights/qsr-innovation",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/impact-strategy-and-reporting/stakeholder-engagement-public-policy.html",
"https://corporate.mcdonalds.com/corpmcd/about-us/our-history.html",
"https://corporate.mcdonalds.com/corpmcd/investor-relations/annual-reports.html",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://www.ukessays.com/essays/management/analysis-of-mcdonalds-leadership-style-management-essay.php",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://www.researchgate.net/publication/237625918_The_leadership_of_Ronald_McDonald_Double_narration_and_stylistic_lines_of_transformation",
"https://corporate.mcdonalds.com/corpmcd/our-people/education-programs.html",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/operations.html",
"https://www.qsrmagazine.com/workforce/why-better-leadership-keeps-employees",
"https://trainingmag.com/mcdonalds-leadership-journey/",
"https://www.bbc.com/news/business-52483647",
"https://www.businessinsider.com/mcdonalds-franchise-employees-culture-complaints-2022-06",
"https://hbr.org/2020/12/what-your-youngest-workers-need-most-right-now",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/employee-engagement.html",
"https://corporate.mcdonalds.com/corpmcd/our-people/education-programs.html",
"https://www.linkedin.com/pulse/your-leadership-glass-half-full-chris-kempczinski-qixic",
"https://www.fastcompany.com/90809050/what-gen-z-really-wants-at-work",
"https://www.businessinsider.com/mcdonalds-franchise-employees-culture-complaints-2022-06",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://hbr.org/2023/05/employee-ideas-fast-food",
"https://corporate.mcdonalds.com/corpmcd/investor-relations/annual-reports.html",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/commitment-to-inclusion.html",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/safety.html",
"https://www.cnbc.com/2023/01/26/mcdonalds-bonuses-tied-to-dei-progress.html",
"https://www.hrmagazine.co.uk/content/news/mcdonald-s-uk-launches-community-champions-programme",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/employee-engagement.html",
"https://www.linkedin.com/pulse/mcdonalds-theory-leadership-employee-engagement-robert-schimmenti",
"https://corporate.mcdonalds.com/corpmcd/our-people/education-programs.html",
"https://ivypanda.com/essays/mcdonalds-company-employee-engagement/",
"https://www.hrmagazine.co.uk/content/news/mcdonald-s-uk-launches-community-champions-programme",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://www.vaia.com/en-us/explanations/business-studies/business-case-studies/mcdonalds-organisational-structure/",
"https://www.forbes.com/sites/blakemorgan/2022/03/21/why-customer-experience-is-the-new-brand-currency/",
"https://www.diversityinc.com/mcdonalds-unveils-new-diversity-equity-and-inclusion-initiatives/",
"https://corporate.mcdonalds.com/corpmcd/our-people/franchise.html",
"https://www.linkedin.com/pulse/leadership-fast-food-hr-insights/",
"https://www.glassdoor.com/employers/blog/mcdonalds-employee-reviews/",
"https://corporate.mcdonalds.com/corpmcd/investor-relations/annual-reports.html",
"https://www.forbes.com/sites/forbestechcouncil/2023/03/13/tech-rollouts-and-employee-fatigue-in-qsr/",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/safety.html",
"https://corporate.mcdonalds.com/corpmcd/our-stories/article/global-standards.html",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/jobs-inclusion-and-empowerment/talent-and-benefits.html",
"https://www.linkedin.com/pulse/mcdonalds-theory-leadership-employee-engagement-robert-schimmenti",
"https://www.ericgoldman.name/en/2008/observation-of-leadership-and-organizational-behavior-at-mcdonalds/",
"https://www.cnbc.com/2024/02/10/fast-food-turnover-costs.html",
"https://hbr.org/2023/07/franchise-leadership-stress",
"https://www.fastcompany.com/90809050/gen-z-feedback",
"https://www.forbes.com/sites/forbestechcouncil/2024/01/ai-adoption-challenges",
"https://www.bbc.com/news/business-50214217",
"https://www.glassdoor.com/Reviews/McDonald-s-Manager-Hours",
"https://www.osha.gov/mcdonalds-compliance",
"https://www.mckinsey.com/industries/retail/our-insights/qsr-innovation",
"https://corporate.mcdonalds.com/corpmcd/our-purpose-and-impact/impact-strategy-and-reporting/stakeholder-engagement-public-policy.html",
"https://www.businessinsider.com/mcdonalds-franchise-employees-culture-complaints-2022-06",
"https://hbr.org/2022/06/why-restaurant-managers-burn-out",
"https://www.fastcompany.com/90809050/what-gen-z-really-wants-at-work",
"https://www.forbes.com/sites/forbestechcouncil/2023/03/13/tech-rollouts-and-employee-fatigue-in-qsr/"
]

# Run the check and automatically save to CSV
results = bulk_check(urls_to_check)

# Print summary to console
print("\nSummary Results:")
print(f"Total URLs checked: {len(results)}")
print(f"Good URLs: {sum(1 for r in results if r['Valid'])}")
print(f"Bad URLs: {sum(1 for r in results if not r['Valid'])}")
#bulk_check(urls_to_check)