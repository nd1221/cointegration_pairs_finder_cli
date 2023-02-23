import scrapy


class ForexScraper(scrapy.Spider):
    """Scrapes Central Charts for forex pairs tickers."""
    name = 'forex'
    start_urls= [
        'https://www.centralcharts.com/en/price-list-ranking/ALL/asc/ts_48-forex-128-currency-pairs--qc_1-alphabetical-order',
        'https://www.centralcharts.com/en/price-list-ranking/ALL/asc/ts_48-forex-128-currency-pairs--qc_1-alphabetical-order?p=2',
        'https://www.centralcharts.com/en/price-list-ranking/ALL/asc/ts_48-forex-128-currency-pairs--qc_1-alphabetical-order?p=3',
        'https://www.centralcharts.com/en/price-list-ranking/ALL/asc/ts_48-forex-128-currency-pairs--qc_1-alphabetical-order?p=4',
        'https://www.centralcharts.com/en/price-list-ranking/ALL/asc/ts_48-forex-128-currency-pairs--qc_1-alphabetical-order?p=5',
    ]


    def parse(self, response):
        for pair in response.xpath("//table[@class='tabMini tabQuotes']/tbody/tr"):
            yield {
                'forex_pair': pair.css('td a::text').get(),
                'ticker': f"{pair.css('td a::text').get().replace('/', '')}=X",
            }