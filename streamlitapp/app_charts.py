import pandas as pd
import numpy as np
import altair as alt
import json


# ingredient list
def ingredient_list(ings):
    '''
    Takes in an ingredient list provided by the title to ingredient model
    then returns an altair chart to display on webapp

    inputs:
    ings - list of strings

    returns:
    ingredient_list - altair mark_text chart
    '''

    ingdf = pd.DataFrame(ings)
    ingdf.rename(columns={0:'ingredient'}, inplace=True)
    ingdf['y'] = range(ingdf.shape[0])
    
    # Find common pairings
    ingdf['common pairings'] = ''
    with open('models/common_pairs.json','r') as f:
        most_common_pairs = json.load(f)
    for i,ing in enumerate(ings):
        ing_pairs = [x for x in most_common_pairs if ing in x]
        pairings = [x if x != ing else y for x,y in ing_pairs]
        res = [i for n, i in enumerate(pairings) if i not in pairings[:n]][:5] # remove duplicates and take top 5
        ingdf.at[i,'common pairings'] = ', '.join(res)

    height = len(ings)*30
    width = len(max(ings))*20
    ingredient_list = alt.Chart(ingdf).mark_text(fontSize=20, fontWeight='bold', align='center', color='gray').encode(
        text = 'ingredient:N',
        y = alt.Y('y:O', axis=None),
        tooltip = ['common pairings:N']
    ).configure_axis(grid=False).configure_view(strokeWidth=0).properties(
        height=height,
        width=width
    )
    return ingredient_list

# pie chart
def pie_chart(cuisines):
    cuisinedf = pd.DataFrame(cuisines, index=[0]).T
    cuisinedf.reset_index(inplace=True)
    cuisinedf.rename(columns={'index':'ethnicity',0:'percent'}, inplace=True)
    piechart = alt.Chart(cuisinedf).mark_arc(radius=110).encode(
        theta = alt.Theta('percent:Q',stack=True), 
        order = alt.Order('percent:Q', sort='descending'),
        color = alt.Color('ethnicity:N', sort=np.array(cuisinedf.ethnicity.values), legend=None), 
        tooltip = ['ethnicity:N', 'percent:Q']
    )
    text = piechart.mark_text(radius=150).encode(
        text=alt.condition('datum.percent >= 5','ethnicity:N',alt.value(''))
        )
    perctext = piechart.mark_text(radius=90).encode(
        text=alt.condition('datum.percent >= 5','percent:Q',alt.value('')),
        color=alt.value('black')
        )
    return piechart+text+perctext

# t-SNE
def tSNE_chart(df,recipedf):
    # selections
    mouse = alt.selection_single(on='mouseover',fields=['label'])    
    legend = alt.selection_multi(fields=['label'], bind='legend')

    # main t-SNE chart
    c = alt.Chart(df).mark_circle(size=60,stroke='black', strokeWidth=.25).encode(
            x = alt.X('x:Q', axis=None),
            y = alt.Y('y:Q', axis=None),
            color = alt.condition(legend | mouse,'label:N',alt.value('gray')),#'label:N',
            #tooltip = ['word','label',],
            )        
    t = c.mark_text(dy=8, fontSize=10).encode(
        text=alt.condition(legend | mouse,'word:N',alt.value('')),
        )

    # predicted recipe point
    recipe = alt.Chart(recipedf).mark_point(fill='white',stroke='gray').encode(
            x = 'x:Q',
            y = 'y:Q',
            # color = alt.value('white'),
            shape = alt.value('diamond'),
            # size = alt.Size('distinction:N', scale=alt.Scale(range=[15,30]))
            #tooltip = ['word','label',],
            )
    t_r = recipe.mark_text(dy=-15,fontWeight='bold',fill='white',stroke='black',strokeWidth=.85).encode(
        text=alt.Text('label:N'),
        size = alt.Size('distinction:N', scale=alt.Scale(range=[15,25]), legend=None)
    )

    # combine and display        
    combined = (c+t+recipe+t_r).interactive().add_selection(
            mouse, legend
            ).configure_axis(
            grid=False
            ).configure_legend(
                labelFontSize=18,
                orient='left',
            ).configure_view(
                strokeWidth = 0
            ).properties(
                height=600,
                width = 700
            )

    c_notext = alt.Chart(df).mark_circle(size=120).encode(
            x = alt.X('x:Q', axis=None),
            y = alt.Y('y:Q', axis=None),
            color = alt.condition(legend | mouse,'label:N',alt.value('gray')),#'label:N',
            #tooltip = ['word','label',],
            )        
    t = c.mark_text(dy=8, fontSize=10).encode(
        text=alt.condition(legend | mouse,'word:N',alt.value('')),
        )

    combined_notext = (c_notext+recipe+t_r).interactive().add_selection(
        mouse, legend
        ).configure_axis(
        grid=False
        ).configure_legend(
            labelFontSize=18,
            orient='left',
        ).properties(
            height=600,
            width = 1000
        )
    
    combined.save('_static/tsne.html')

    return combined, combined_notext