import scrapy


class ftse_spider(scrapy.Spider):
    """FTSE scraper."""
    name = 'ftse_spider'
    start_urls = [
        'https://en.wikipedia.org/wiki/FTSE_100_Index',
        'https://en.wikipedia.org/wiki/FTSE_250_Index',
        ]


    def parse(self, response):
        # Get index name.
        index = response.css('h1 span::text').get()

        # Get stock, ticker and industry.
        # Add exchange and index; then yield. 
        for stock in response.xpath("//table[@id='constituents']/tbody/tr")[1:]:
            yield {
                'security': stock.css('td a::text').get().strip(),
                'ticker': stock.css('td::text').getall()[0].strip(),
                'index': index,
            }
