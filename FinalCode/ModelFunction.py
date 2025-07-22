# MODELFUNCTION

# Lunar Leaper: Gravimetry Modelling
# Yara Luginbühl


import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.gravimetry import GravityModelling2D, solveGravimetry
import matplotlib.pyplot as plt


def g_korr(x1,x2,y1,y2,x,y, rho):
    G = 6.6743e-11 #m3 kg-1 s-2
 
    theta1 = np.arctan((x2-x)/(y2-y))
    theta2 = np.arctan((x1-x)/(y2-y))
    theta3 = np.arctan((x2-x)/(y1-y))
    theta4 = np.arctan((x1-x)/(y1-y))
    return 2*G*rho*((y2-y)*(theta1 - theta2) - (y1-y)*(theta3 - theta4))*1e5

def g_korr_x(x1,x2,y1,y2,x,y, rho):
    G = 6.6743e-11 #m3 kg-1 s-2

    theta1 = np.arctan((y2-y)/(x2-x))
    theta2 = np.arctan((y1-y)/(x2-x))
    theta3 = np.arctan((y2-y)/(x1-x))
    theta4 = np.arctan((y1-y)/(x1-x))
    return 2*G*rho*((x2-x)*(theta1 - theta2) - (x1-x)*(theta3 - theta4))*1e5

def LavaTubeGravimetry(tube_radius, tube_depth, regolith_thickness):

    # INPUT: 
    
    # tube_radius: radius of the lava tube in meters
    # tube_depth: depth of the lava tube in meters  
    # regolith_thickness: thickness of the regolith layer in meters

    # OUTPUT: 

    # points_x: x-coordinates of the measurement points
    # g_final: gravity field at the measurement points without corrections
    # g_corr: gravity field at the measurement points with corrections
    # Model: {
    #     'Geom': geometry of the model,  
    #     'Mesh': mesh of the model,
    #     'dRho': density array for the mesh
    # }


    # DEFINE GEOMETRY: 

    # Define rille variables:
    x_top = 210
    x_bottom = 210
    y_top = 0  - regolith_thickness
    y_bottom = -10 - regolith_thickness

    # World edges
    xmax = 1500
    ymin = -300

    # lava tube
    lava_tube_center_depth = tube_depth
    lava_tube_diameter = 2*tube_radius
    lava_tube_height = 30

    world = mt.createPolygon(
        [
            (-xmax, y_top),   
            (-x_top, y_top), 
            (-x_bottom, y_bottom),  
            (x_bottom, y_bottom),   

            (x_top, y_top),   
            (xmax, y_top),    
            (xmax, ymin),  
            (-xmax, ymin)  
        ],
        isClosed=True,
        addNodes=3,
        marker=1,
        boundaryMarker=1
    )

    cave = mt.createCircle(pos=[-0, -lava_tube_center_depth], 
                        radius=[lava_tube_diameter / 2, lava_tube_height / 2], marker=2,
                        boundaryMarker=10, area=100)


    regolith_y_top = y_top + regolith_thickness
    regolith_y_bottom = y_top 
    regolith_rille_y_top = y_bottom + regolith_thickness
    regolith_rille_y_bottom = y_bottom


    # Create regolith polygon (covers full top area)
    regolith_l = mt.createPolygon([
        (-xmax, regolith_y_top),
        (-x_top, regolith_y_top),
        (-x_top, regolith_y_bottom),
        (-xmax, regolith_y_bottom)
    ], isClosed=True, addNodes=3, marker=3)

    regolith_r = mt.createPolygon([
        (xmax, regolith_y_top),
        (x_top, regolith_y_top),
        (x_top, regolith_y_bottom),
        (xmax, regolith_y_bottom)
    ], isClosed=True, addNodes=3, marker=3)

    regolith_m = mt.createPolygon([
        (-x_top, regolith_rille_y_top),
        (x_top, regolith_rille_y_top),
        (x_top, regolith_rille_y_bottom),
        (-x_top, regolith_rille_y_bottom)
    ], isClosed=True, addNodes=3, marker=3)

    y_rille = regolith_rille_y_top
    pit_R = 25
    pit_bottom = -(lava_tube_center_depth - lava_tube_height/2) 


    # pit = mt.createPolygon([
    #     (-pit_R, regolith_rille_y_top),
    #     (pit_R, regolith_rille_y_top),
    #     (pit_R, pit_bottom),
    #     (-pit_R, pit_bottom)
    # ], isClosed=True, addNodes=2, marker=2)



    regolith = regolith_l + regolith_r + regolith_m 

    if (regolith_thickness == 0):
        geom = world + cave
    else: 
        geom = world + cave + regolith

    mesh = mt.createMesh(geom, quality=33, area=100.)


    # DEFINE DENSITIES
    layer_densities = [[1, 2500], # region marker, density
                    [2, 0], 
                    [3, 1500]]

    dRho = pg.solver.parseMapToCellArray(layer_densities, mesh) # map layer densities to mesh 

    # DEFINE MEASUREMENT POSITIONS
    top_edges = []

    for b in geom.boundaries():
        x = b.center()[0]
        y = b.center()[1]

        if (
            (abs(b.center()[0]) <= x_top and b.center()[1] >= y_rille)
            or
            (xmax > abs(b.center()[0]) > x_top) and (b.center()[1] >= regolith_y_top)
        ):
            top_edges.append((x,y))

    top_edges = sorted(top_edges, key=lambda pt: (pt[0], pt[1] if pt[0] > 0 else -pt[1]))


    top_edges_x = [p[0] for p in top_edges]
    top_edges_y = [p[1] for p in top_edges]

    delta_s = 5
    points_x = np.arange(top_edges_x[0], top_edges_x[-1]+1, delta_s)
    points_y = np.interp(points_x, top_edges_x, top_edges_y)
    pnts = np.array([points_x, points_y]).T


    world_edges = [
        (c.center()[0], c.center()[1]) for c in world.boundaries()
        if abs(c.center()[0]) < xmax and c.center()[1] >= y_bottom
    ]
    world_edges_x = [p[0] for p in world_edges]
    world_edges_y = [p[1] for p in world_edges]

    world_x = np.arange(world_edges_x[0], world_edges_x[-1]+1, delta_s)
    world_y = np.interp(world_x, world_edges_x, world_edges_y)
    world_pnts = np.array([world_x, world_y])

    # plt.figure(figsize = (20,3))
    # plt.title("Measurements")
    # plt.plot(top_edges_x, top_edges_y, "ko", markersize = 4, label = "boundary")
    # plt.plot(world_edges_x, world_edges_y, "ro", markersize = 4, label = "boundary w/o regolith")
    # plt.plot(points_x, points_y, "k+", markersize = 4, label = "Measurements")
    # plt.plot(world_x, world_y, "r+", markersize = 4, label = "Interp w/o regolith")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # # plt.ylim(-10, 5)
    # plt.show()
    # plt.savefig("FirstModel_V2_GridAndMeasurements")
 
    # GRAVIMETRY MODELLING
    print("Begin modeling")
    fop = GravityModelling2D(mesh=mesh, points=pnts)
    g_vector = solveGravimetry(fop.regionManager().paraDomain(), # output: gravity field measured at each measuring point (len(points))
                            dRho,
                            pnts=fop.sensorPositions,
                            complete=True)
    print("End modeling")

    g = g_vector[0]
    g_x = g[:,0]
    g_z = g[:,2]

    # CORRECTIONS:
    # Free air correction: 
    g_mond = 1.62 #m/s^2
    R_mond = 1738e3 #m
    free_air_correction = ((2*g_mond/R_mond) * points_y)*1e5 #mGal

    # Bouguer anomaly:
    G = 6.6743e-11 #m3 kg-1 s-2
    rho = 2500 # kg/m³
    rho_regolith = 1500 # kg/m³
    bouguer = rho*(2*np.pi*G*(world_y))*1e5 + rho_regolith*(2*np.pi*G*(points_y - world_y))*1e5 #mGal

    # Terrain correction (z-Axis)
    g_terrain = np.zeros(len(points_x))
    g_regolith = np.zeros(len(points_x))

    for i,x in enumerate(points_x):
        y = points_y[i] 

        #block 1 (x1,x2,y1,y2)
        x1_1 = -xmax
        x2_1 = -x_top
        y1_1 = regolith_rille_y_top
        y2_1 = y_top

        if (-x_top<= x <= x_top) or (x > x_top):
            g_terrain[i] += g_korr(x1_1, x2_1, y1_1, y2_1, x,y, rho)


        # block 1 regolith
        x1_regolith_l = -xmax
        x2_regolith_l = -x_top
        y1_regolith_l = y_top
        y2_regolith_l = regolith_y_top

        if (-x_top<= x <= x_top) or (x > x_top):
            g_regolith[i] += g_korr(x1_regolith_l, x2_regolith_l, y1_regolith_l, y2_regolith_l, x,y, rho_regolith)

    
        #block 2
        x1_2 = -x_top
        x2_2 = x_top
        y1_2 = regolith_rille_y_top
        y2_2 = y_top
        
        if (x < -x_top) or (x > x_top):
            g_terrain[i] -= g_korr(x1_2, x2_2, y1_2, y2_2, x, y, rho)
        
        # block 2 regolith
        x1_regolith_m = -x_top
        x2_regolith_m = x_top
        y1_regolith_m = y_top
        y2_regolith_m = regolith_y_top

        if (x < -x_top) or (x > x_top):
            g_regolith[i] -= g_korr(x1_regolith_m, x2_regolith_m, y1_regolith_m, y2_regolith_m, x, y, rho_regolith)
        

        #block 3
        x1_3 = x_top
        x2_3 = xmax
        y1_3 = regolith_rille_y_top
        y2_3 = y_top
        
        if (-x_top<= x <= x_top) or (x < x_top):
            g_terrain[i] += g_korr(x1_3, x2_3, y1_3, y2_3, x, y, rho)

        # block 3 regolith
        x1_regolith_r = x_top
        x2_regolith_r = xmax    
        y1_regolith_r = y_top
        y2_regolith_r = regolith_y_top

        if (-x_top<= x <= x_top) or (x < x_top):
            g_regolith[i] += g_korr(x1_regolith_r, x2_regolith_r, y1_regolith_r, y2_regolith_r, x, y, rho_regolith)


    g_z_corr = g_z + bouguer - free_air_correction + g_terrain + g_regolith 

 
    # Terrain correction (x-axis)

    g_terrain_x = np.zeros(len(points_x))
    g_regolith_x = np.zeros(len(points_x))


    for i,x in enumerate(points_x):
        y = points_y[i] 

        #block 1 (x1,x2,y1,y2)
        x1_1 = -xmax
        x2_1 = -x_top
        y1_1 = regolith_rille_y_top
        y2_1 = y_top
        
        if (-x_top<= x <= x_top) or (x > x_top):
            g_terrain_x[i] += g_korr_x(x1_1, x2_1, y1_1, y2_1, x,y, rho)


        # block 1 regolith
        x1_regolith_l = -xmax
        x2_regolith_l = -x_top
        y1_regolith_l = y_top
        y2_regolith_l = regolith_y_top

        if (-x_top<= x <= x_top):# or (x > x_top):
            g_regolith_x[i] += g_korr_x(x1_regolith_l, x2_regolith_l, y1_regolith_l, y2_regolith_l, x,y, rho_regolith)

    
        #block 2
        x1_2 = -x_top
        x2_2 = x_top
        y1_2 = regolith_rille_y_top
        y2_2 = y_top
        
        if (x < -x_top) or (x > x_top):
           g_terrain_x[i] -= g_korr_x(x1_2, x2_2, y1_2, y2_2, x, y, rho)
        
        # block 2 regolith
        x1_regolith_m = -x_top
        x2_regolith_m = x_top
        y1_regolith_m = y_top
        y2_regolith_m = regolith_y_top

        if (x < -x_top) or (x > x_top):
           g_regolith_x[i] -= g_korr_x(x1_regolith_m, x2_regolith_m, y1_regolith_m, y2_regolith_m, x, y, rho_regolith)
        

        #block 3
        x1_3 = x_top
        x2_3 = xmax
        y1_3 = regolith_rille_y_top
        y2_3 = y_top
        
        if (-x_top<= x <= x_top): #or (x < x_top):
            g_terrain_x[i] += g_korr_x(x1_3, x2_3, y1_3, y2_3, x, y, rho)

        # block 3 regolith
        x1_regolith_r = x_top
        x2_regolith_r = xmax    
        y1_regolith_r = y_top
        y2_regolith_r = regolith_y_top

        if (-x_top<= x <= x_top): #or (x < x_top):
            g_regolith_x[i] += g_korr_x(x1_regolith_r, x2_regolith_r, y1_regolith_r, y2_regolith_r, x, y, rho_regolith)


    g_x_corr = g_x - g_terrain_x - g_regolith_x

    g_final = np.sqrt(g_x**2 + g_z**2)
    g_corr = np.sqrt(g_x_corr**2 + g_z_corr**2)

    Model = {
        'Geom': geom,
        'Mesh': mesh,
        'dRho': dRho
    }
    return points_x, g_final, g_corr, Model


