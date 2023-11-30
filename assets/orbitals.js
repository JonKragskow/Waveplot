window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        orbital_function: function(orbital, orb_iso, colour_a, colour_b, wire_toggle,
            orb_name_3d, x, y, z, zoom, qx, qy, qz, qw) {

    let element = document.getElementById('orb_mol_div');

    while(element.firstChild){
        element.removeChild(element.firstChild);
    }

    let config = { backgroundOpacity: 0.};
    let viewer = $3Dmol.createViewer(element, config);

    viewer.getCanvas()['style']='width: 100vw;'
    viewer.getCanvas()['id']='viewer_canvas'

    var m = viewer.addModel();

    var voldata = new $3Dmol.VolumeData(orbital, 'cube');
    viewer.addIsosurface(
        voldata,
        {
            isoval: orb_iso,
            color: colour_a,
            wireframe: wire_toggle,
            smoothness: 10
        }
    );
    viewer.addIsosurface(
        voldata,
        {
            isoval: -orb_iso,
            color: colour_b,
            wireframe: wire_toggle,
            smoothness: 10
        }
    );

    viewer.render();
    viewer.center();

    if (orb_iso !== 0) {
        if (document.getElementById('orb_view_zoom').value == ''){
            viewer.zoomTo();
            zoom_level = viewer.getView()[3]
        }
        else {
            zoom_level = parseFloat(document.getElementById('orb_view_zoom').value)
        }

        console.log(parseFloat(document.getElementById('orb_view_x').value))
        viewer.setView([
            parseFloat(document.getElementById('orb_view_x').value),
            parseFloat(document.getElementById('orb_view_y').value),
            parseFloat(document.getElementById('orb_view_z').value),
            zoom_level,
            parseFloat(document.getElementById('orb_view_qx').value),
            parseFloat(document.getElementById('orb_view_qy').value),
            parseFloat(document.getElementById('orb_view_qz').value),
            parseFloat(document.getElementById('orb_view_qw').value)
        ])
        viewer.getCanvas().addEventListener('wheel', (event) => { updateViewText('orb', viewer) }, false)
        viewer.getCanvas().addEventListener('mouseup', (event) => { updateViewText('orb', viewer) }, false)   
        viewer.getCanvas().addEventListener('touchend', (event) => { updateViewText('orb', viewer) }, false)   
        return zoom_level;
    }
    else {
        return '';
    }

    }}})