import scrapy


class NasdaqScraper(scrapy.Spider):
    """Scrapes ADVFN website for NASDAQ tickers."""
    name = 'nasdaq'
    start_urls = [
        'https://www.advfn.com/nasdaq/nasdaq.asp',
    ]


    def parse(self, response):
        for link in response.xpath("//div[@id='az']/a/@href"):
            yield response.follow(link.get(), callback=self.parse_alphabet)
    

    def parse_alphabet(self, response):
        data = response.css('tr.ts0') + response.css('tr.ts1')
        
        for security in data:
            yield {
                'security': security.css('td a::text')[0].get(),
                'ticker': security.css('td a::text')[1].get(),
            }