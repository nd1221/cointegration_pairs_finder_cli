import scrapy

class Sp500Spider(scrapy.Spider):
    """S&P 500 scraper."""
    name = 'sp500_spider'
    start_urls = [
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    ]


    def parse(self, response):
        # Get index name.
        indices = ['S&P 500']
        for index in indices:
            if index in response.css('h1 span::text').get():
                index = index

        # Get security names and ticker symbols, then yield.
        for stock in response.xpath("//table[@id='constituents']/tbody/tr")[1:]:
            yield {
                'security': stock.css('td a::text').getall()[1],
                'ticker': stock.css('td a::text').get(),
                'index': index, 
            }