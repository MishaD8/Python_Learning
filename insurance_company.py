print('Insurance Company Ad Creator')

company_name = input('What is the name of your company  ')
num_of_employees = input('Number of employees  ')
location = input('Country of Origin  ')
priceL = input('Lowest Policy Price  ')
priceH = input('Highest Policy Price  ')

print(f"""We are {company_name} located in {location}. Our {num_of_employees} employees can help 
you find the policy that is right for you with policies ranging from ${priceL} to ${priceH} 
per month! """)