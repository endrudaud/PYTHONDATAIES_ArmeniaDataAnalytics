import ipywidgets as widgets
from IPython.display import display, Markdown
from IPython.core.display import HTML

# Load the custom CSS file
display(HTML("<style>{}</style>".format(open("style.css").read())))
def setup_display():
    # No special setup required
    pass

def create_tabs_for_sections(sections):
    # Create a Tab widget
    tab = widgets.Tab()

    # Store content and titles for the tabs
    children = []
    titles = []

    for section in sections:
        with open(f"sections/{section}.md", "r") as file:
            content = file.read()
        # Create an Output widget for each section and add it to the list
        out = widgets.Output()
        with out:
            display(Markdown(content))
        children.append(out)
        titles.append(section)
    
    # Set the children and titles for the tabs
    tab.children = children
    for i in range(len(titles)):
        tab.set_title(i, titles[i])

    # Display the Tab widget
    display(tab)