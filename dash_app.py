from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash

def create_dash_app():
    app = JupyterDash(__name__)

    app.layout = html.Div([
        html.H1("Armenian Population Perception Analysis"),
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Dissolution of Soviet Union', value='tab-1'),
            dcc.Tab(label="Armenia's Main Enemy", value='tab-2'),
            dcc.Tab(label="Armenia's Main Friend", value='tab-3'),
            dcc.Tab(label='Opinion about EU', value='tab-4'),
            dcc.Tab(label='Summary and Future Outlook', value='tab-5')
        ]),
        html.Div(id='tabs-content')
    ])

    @app.callback(
        Output('tabs-content', 'children'),
        [Input('tabs', 'value')]
    )
    def render_content(tab):
        if tab == 'tab-1':
            return html.Div([
                html.H3('Perception of the Dissolution of the Soviet Union'),
                html.P("""
                    Contrary to our expectations, there was no significant correlation between respondents' political affiliations or income levels and their views on the dissolution of the Soviet Union.
                    However, education and age played a significant role: respondents with higher education levels were more likely to view the dissolution positively, as were younger respondents, indicating a generational shift in perspectives.
                """),
            ])
        elif tab == 'tab-2':
            return html.Div([
                html.H3("Perception of Armenia's Main Enemy"),
                html.P("""
                    The analysis showed that respondents who support EU integration were more likely to perceive Russia as Armenia's main enemy, while those who view Russia as an ally tended to see Azerbaijan as the primary enemy.
                    Contrary to our expectations, political party affiliation did not significantly influence perceptions of Turkey as Armenia's main enemy, suggesting that Armenian political parties may not be effectively shaping public opinion on this issue.
                    Additionally, education level did not significantly impact the perception of Russia as Armenia's main enemy.
                """),
            ])
        elif tab == 'tab-3':
            return html.Div([
                html.H3("Perception of Armenia's Main Friend"),
                html.P("""
                    The perception of Russia as Armenia's main friend did not differ significantly between older and younger generations, counter to our initial assumptions as the older generation was inclined to believe that the resolution of the USSR was bad.
                    This indicates that people may have a mistaken notion about the USSR and do not perceive it as a Russian communist empire.
                    However, this is a topic for further research. Respondents who identified Turkey or Azerbaijan as Armenia's main enemy were more likely to consider Russia a friend, highlighting the complex geopolitical dynamics at play.
                """),
            ])
        elif tab == 'tab-4':
            return html.Div([
                html.H3("Opinion about the European Union (EU)"),
                html.P("""
                    Surprisingly, none of the hypothesized factors—age, education, or religiosity—significantly influenced respondents' opinions about the EU.
                    This suggests that public opinion on the EU is less affected by demographic or socio-cultural factors than by other, perhaps more immediate, concerns. This will also require further research.
                """),
            ])
        elif tab == 'tab-5':
            return html.Div([
                html.H3("Summary and Future Outlook"),
                html.P("""
                    Our findings indicate that traditional predictors such as political affiliation and income levels have limited influence on certain perceptions within the Armenian population, particularly regarding the dissolution of the Soviet Union and views on the EU.
                    In contrast, education and age were found to significantly influence opinions on the dissolution, reflecting broader societal shifts.
                """),
                html.P("""
                    To sum up, it is important to note that the data was collected in 2019, prior to the Second Nagorno-Karabakh War and the subsequent ethnic cleansing and mass migration of Armenians from Nagorno-Karabakh—events that have significantly impacted public opinion.
                    Preliminary reports and limited surveys suggest a dramatic shift in perceptions. For instance, a significant portion of the population now views Russia as a major threat and France as Armenia's main ally.
                    Comprehensive data from the latest Caucasus Barometer survey, expected next year, will provide data for 2024, and we plan to compare these findings to our current analysis to understand how recent events have reshaped Armenian public opinion.
                """),
            ])

    return app