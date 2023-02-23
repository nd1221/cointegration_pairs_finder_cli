import scrapy


class SecurityScraper(scrapy.Spider):
    """Scrapes ADVFN website."""
    name = 'security_spider'
    start_urls = [
        'https://www.advfn.com/nyse/newyorkstockexchange.asp?companies=A',
    ]


    def parse(self, response):
        for security in response.css('tr.ts0'):
            yield {
                'security': security.css('td a::text').getall()[0],
                'ticker': security.css('td a::text').getall()[1],
            }
        
        