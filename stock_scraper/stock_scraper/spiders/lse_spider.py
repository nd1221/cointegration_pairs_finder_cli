import scrapy


class LseScraper(scrapy.Spider):
    """Scrapes ADVFN website for LSE tickers."""
    name = 'lse'
    start_urls = [
        'https://uk.advfn.com/stock-market/london',
    ]


    def parse(self, response):
        for link in response.xpath("//div[@id='a-to-z']/ul/li/a/@href"):
            yield response.follow(link.get(), callback=self.parse_alphabet)

    
    def parse_alphabet(self, response):
        names = response.css('td.String.Column1 a::text').getall()
        tickers = response.css('td.String.Column2.ColumnLast::text').getall()
        
        for i in range(len(names)):
            yield {
                'security': names[i],
                'ticker': f"{tickers[i]}.L",
            }