import scrapy


class AmexScraper(scrapy.Spider):
    """Scrapes ADVFN website for AMEX tickers."""
    name = 'amex'
    start_urls = [
        'https://www.advfn.com/amex/americanstockexchange.asp',
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
