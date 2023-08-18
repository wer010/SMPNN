import numpy as np
import plotly.graph_objects as go


def plot_3d_graph(points, edge_index):
    '''
    use plotly to visualize the 3d graph
    :param points: 3d point, which shape is [n,3]
    :param edge_index: edge index, which shape is [2,m], keep the same format with PYG
    :return:
    usage:  points = np.random.random([10,3])
            edge_index = np.array([[0,0,0,0,0],[1,2,3,4,5]])
            plot_3d_graph(points,edge_index)
    '''

    Xe = []
    Ye = []
    Ze = []
    for e in edge_index.T:
        Xe += [points[e[0]][0], points[e[1]][0]]  # x-coordinates of edge ends
        Ye += [points[e[0]][1], points[e[1]][1]]
        Ze += [points[e[0]][2], points[e[1]][2]]

    fig = go.Figure()

    vertices =go.Scatter3d(
              x=points[:,0],
              y=points[:,1],
              z=points[:,2],
              mode='markers',
              name='vertices',
              marker=dict(symbol='circle',
              size=1,
              color='rgb(150,150,150)',
              colorscale='Viridis',
              line=dict(color='rgb(50,50,50)', width=0.5)),
              text=np.arange(points.shape[0]),
              hoverinfo='text')

    edges = go.Scatter3d(x = Xe,
                         y = Ye,
                         z = Ze,
                         mode='lines',
                         line=dict(color='rgb(125,125,125)', width=1),
                         name='edges',
                         hoverinfo='none')

    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title='')
    layout = go.Layout(
             title="(3D visualization)",
             width=1000,
             height=1000,
            #  showlegend=False,

         margin=dict(
            t=100
        ),
        hovermode='closest',
        )

    fig.add_trace(vertices)
    fig.add_trace(edges)

    fig.update_layout(layout)
    fig.show()



