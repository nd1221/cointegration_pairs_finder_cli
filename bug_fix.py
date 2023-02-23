# Nevermind, bug fixed
# =====================================================================================================================

# Fix a bug I created when writing unit tests
# If the pickle file is empty save function in find_pair() will not execute without being aborted by click

class InitialPickleObject():
    
    def __init__(self):
        self.name = 'SAVED COINTEGRATEDPAIR OBJECTS'