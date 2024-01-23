import dash

dash.register_page(
	__name__,
	title='Forward Outlook',
	description='This is the forward outlook',  # should accept callable too
	order=6,
	path='/forward-outlook',
    image='birds.jpeg'
)


def layout():
	return 'Forward outlook'