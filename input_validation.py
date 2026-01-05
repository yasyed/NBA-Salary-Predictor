
#Validate responses to the feature input. Ensure users are only inputting data that makes sense for model
def validate_input (question, answer_type, min_value = 0, max_value = None):
    while True:
        try:
            answer = answer_type(input(question))
            if answer < min_value:
                print(f'This is invalid input, enter a value greater than {min_value}')
                continue
            if answer > max_value:
                print(f'This is invalid input, enter a value less than {max_value}')
                continue
            return answer
        except ValueError:
            print("This is invalid input, please try again")

#Validated general navigation through app. Users get error message if they press a non-existent path number
def validate_navigation_input (number_options):
    while True:
        try:
            answer = int(input())
            if answer < 1:
                print(f'This is invalid input, enter a valid option')
                continue
            if answer > number_options:
                print(f'This is invalid input, enter a valid option')
                continue
            return answer
        except ValueError:
            print("This is invalid input, please try again")