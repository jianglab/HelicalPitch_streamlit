#!/usr/bin/env python

# $Id$

import numpy as np
import pandas as pd
import streamlit as st

def main():
    title = "HelicalPitch: determine helical pitch/twist using 2D Classification info"
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    with st.expander(label="README", expanded=False):
            st.write("This is a Web App to help users determine helical pitch/twist using 2D classification info.  \nNOTE: the uploaded files are **strictly confidential**. The developers of this app do not have access to the files")

    col1, [col2, col3] = st.sidebar, st.columns([0.5,1])

    with col1:
        input_modes_params = {0:"upload", 1:"url"} 
        help_params = "Only RELION star and cryoSPARC cs formats are supported"
        input_mode_params = st.radio(label="How to obtain the Class2D parameter file:", options=list(input_modes_params.keys()), format_func=lambda i:input_modes_params[i], index=1, horizontal=True, help=help_params, key="input_mode_params")
        if input_mode_params == 0: # "upload a star or cs file":
            label = "Upload the class2d parameters in a RELION star or cryoSPARC cs file"
            help = None
            fileobj = st.file_uploader(label, type=['star', "cs"], help=help, key="upload_params")
            if fileobj is not None:
                params = get_class2d_params_from_uploaded_file(fileobj)
            else:
                return
        else:   # download from a URL
            label = "Download URL for a RELION star or cryoSPARC cs file"
            help = None
            url_default = "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/768px/run_it020_data.star"
            url = st.text_input(label, value=url_default, help=help, key="url_params")
            if url is not None:
                params = get_class2d_params_from_url(url)
            else:
                return

        distseg = estimate_inter_segment_distance(params)
        params["segmentid"] = assign_segment_id(params, distseg)

        helices = list(params.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
        nClasses = len(params["rlnClassNumber"].unique())
        st.write(f"{len(params):,} particles | {len(helices):,} filaments | {nClasses} classes")

        required_attrs = np.unique("rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnCoordinateX rlnCoordinateY rlnClassNumber rlnAnglePsi".split())
        missing_attrs = [attr for attr in required_attrs if attr not in params]
        if missing_attrs:
            st.error(f"ERROR: parameters {missing_attrs} are not available")
            st.stop()

        st.divider()

        input_modes_classes = {0:"upload", 1:"url"} 
        help_classes = "Only MRC (*\*.mrcs*) format is supported"
        input_mode_classes = st.radio(label="How to obtain the class average images:", options=list(input_modes_classes.keys()), format_func=lambda i:input_modes_classes[i], index=1, horizontal=True, help=help_classes, key="input_mode_classes")
        if input_mode_classes == 0: # "upload a MRC file":
            label = "Upload the class averages in MRC format (.mrcs, .mrc)"
            help = None
            fileobj = st.file_uploader(label, type=['mrcs'], help=help, key="upload_classes")
            if fileobj is not None:
                data_all, apix = get_class2d_from_uploaded_file(fileobj)
            else:
                return
        else:   # download from a URL
            label = "Download URL for a RELION or cryoSPARC Class2D output mrc(s) file"
            help = None
            url_default = "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/768px/run_it020_classes.mrcs"
            url = st.text_input(label, value=url_default, help=help, key="url_classes")
            if url is not None:
                data_all, apix = get_class2d_from_url(url)
            else:
                return

        n, ny, nx = data_all.shape
        st.write(f"{n} classes: {nx}x{ny} pixels | {round(apix,4)} Å/pixel")

        abundance = np.zeros(n, dtype=int)
        for gn, g in params.groupby("rlnClassNumber"):
            abundance[int(gn)-1] = len(g)
        display_seq = np.arange(n, dtype=int)

        ignore_blank = st.checkbox("Ignore blank classes", value=True, key="ignore_blank")
        sort_abundance = st.checkbox("Sort the classes by abundance", value=True, key="sort_abundance")
        if sort_abundance:
            display_seq = np.argsort(abundance)[::-1]

        with st.expander(label="Choose a class", expanded=True):
            from st_clickable_images import clickable_images
            images = []
            images_displayed = []
            for i in display_seq:
                if abundance[i] or not ignore_blank:
                    images.append(encode_numpy(data_all[i], vflip=True))
                    images_displayed.append(i)
            thumbnail_size = 128
            n_per_row = 400//thumbnail_size
            with st.container(height=min(500, len(images)*thumbnail_size//n_per_row), border=False):
                image_index = clickable_images(
                    images,
                    titles=[f"{i+1} - {abundance[i]}" for i in display_seq if abundance[i] or not ignore_blank],
                    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                    img_style={"margin": "1px", "height": f"{thumbnail_size}px"},
                    key=f"class_display"
                )
                image_index = max(0, image_index)
        class_index = images_displayed[image_index]
        data = data_all[class_index]

        apix_particle = st.number_input("Pixel size of particles", min_value=0.1, max_value=100.0, value=apix, format="%.3f", help="Å/pixel", key="apix_particle")
        
        apix_micrograph_auto = get_pixel_size(params, attrs=["rlnMicrographPixelSize", "rlnMicrographOriginalPixelSize", "rlnImagePixelSize"], return_source=True)
        if apix_micrograph_auto is None:
            apix_micrograph_auto = apix_particle
            apix_micrograph_auto_source = "particle pixel size"
        else:
            apix_micrograph_auto, apix_micrograph_auto_source = apix_micrograph_auto
        apix_micrograph = st.number_input("Pixel size of micrographs for particle extraction", min_value=0.1, max_value=100.0, value=None, placeholder=f"{apix_micrograph_auto:.3f}", format="%.3f", help=f"Å/pixel. Make sure a correct value is specified", key="apix_micrograph")
        if apix_micrograph is None:
            msg = "Please manually input the pixel value for the micrographs used for particle picking/extraction. "
            if apix_micrograph_auto_source in ["rlnMicrographOriginalPixelSize"]:
                msg += f"The placeholder value {apix_micrograph_auto:.3f} was from the **{apix_micrograph_auto_source}** parameter in the input star file. Note that the **{apix_micrograph_auto_source}** value is for the original movies, NOT for the movie averages that were used to extract the particles. If you have binned the movies during motion correction, you should multply {apix_micrograph_auto:.3f} by the bin factor"
                st.warning(msg)
            elif apix_micrograph_auto_source in ["particle pixel size"]:
                msg += f"The placeholder value {apix_micrograph_auto:.3f} was from the **particle image pixel size**. Note that If you have downscaled the particles during particle extraction, you should divide {apix_micrograph_auto:.3f} by the downscaling factor used during particle extraction"
                st.warning(msg)
            st.stop()

    with col2:
        params['rlnCoordinateX'] = params.loc[:, 'rlnCoordinateX'].astype(float)*apix_micrograph
        params['rlnCoordinateY'] = params.loc[:, 'rlnCoordinateY'].astype(float)*apix_micrograph

        if 'rlnOriginXAngst' in params and 'rlnOriginYAngst' in params:
            params.loc[:, 'rlnCoordinateX'] -= params.loc[:, 'rlnOriginXAngst'].astype(float)
            params.loc[:, 'rlnCoordinateY'] -= params.loc[:, 'rlnOriginYAngst'].astype(float)

        particles = params.loc[params["rlnClassNumber"].astype(int)==class_index+1, :]
        helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))

        nHelices = 0
        filement_lengths = []
        for gn, g in helices:
            track_lengths = g["rlnHelicalTrackLengthAngst"].astype(float).values
            l = track_lengths.max() - track_lengths.min() + apix_particle*nx
            filement_lengths.append(l)
            g["filamentLength"] = l
            nHelices += 1

        image_label = f"Class {class_index+1}: {abundance[class_index]:,} segments | {nHelices:,} filaments"
        fig = create_image_figure(data, apix, apix, title=image_label, title_location="above", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=None, show_axis=False, show_toolbar=False, crosshair_color="white")
        st.bokeh_chart(fig, use_container_width=True)
        
        log_y = True
        
        title = f"Filament Lengths: Class {class_index+1}"
        xlabel = "Filament Legnth (Å)"
        ylabel = "# of Filaments"
        fig = plot_histogram(filement_lengths, title=title, xlabel=xlabel, ylabel=ylabel, bins=50, log_y=log_y)
        st.bokeh_chart(fig, use_container_width=True)
        
        min_len = st.number_input("Select filaments of a minimal length (Å)", min_value=0.0, max_value=np.max(filement_lengths), value=0.0, format="%.0f", help="Only use filaments of at least this length for subsequent pair-distance calculation", key="min_len")
        if min_len>0:
            helices_retained = []
            n_ptcls = 0
            for gn, g in helices:
                if g["filamentLength"].values[0]>=min_len:
                    n_ptcls += len(g)
                    helices_retained.append((gn, g))
            st.write(f"{len(helices_retained)} filaments ≥ {min_len} Å in length → {n_ptcls} segments")
        rise = st.number_input("Helical rise (Å)", min_value=0.01, max_value=apix_particle*nx, value=4.75, format="%.2f", help="helical rise", key="rise")

    with col3:
        
        pair_dists = process_one_class(helices)
        title = f"Pair Distances: Class {class_index+1}"
        xlabel = "Pair Distance (Å)"
        ylabel = "# of Pairs"
        fig = plot_histogram(pair_dists, title=title, xlabel=xlabel, ylabel=ylabel, bins=100, log_y=log_y, show_pitch_twist=dict(rise=rise, csyms=(1,2,3)), multi_crosshair=True)
        st.bokeh_chart(fig, use_container_width=True)
        st.write("**How to interpretate the histogram:** an informative histogram should have clear peaks with equal spacing. If so, hover your mouse pointer on the first peak to show the twist values assuming the pair-distance is the helical pitch (adjusted for the cyclic symmetries around the helical axis). If the histogram does not show clear peaks, it indicates that the Class2D quality is bad. You might consider redoing the Class2D task with longer extracted segments (>0.5x helical pitch) from longer filaments (> 1x pitch)")

    return

@st.cache_data(show_spinner=False)
def process_one_class(helices):
    dists_same_class = []

    for hi, helix in enumerate(helices):
        _, segments = helix

        sort_var = ["segmentid"]
        segments = segments.sort_values(sort_var, ascending=True)

        sids = segments["segmentid"].values
        classLabels = segments["rlnClassNumber"].astype(int).values
        loc_x = segments["rlnCoordinateX"].astype(float).values
        loc_y = segments["rlnCoordinateY"].astype(float).values
        psi = segments["rlnAnglePsi"].astype(float).values

        dist = []
        for pi in range(len(sids)):
            ci = classLabels[pi]
            for pj in range(pi + 1, len(sids)):
                cj = classLabels[pj]
                if cj == ci:
                    if abs((psi[pi]-psi[pj]+180) % 360 - 180)<90:   # ensure the same polarity
                        d = np.hypot(loc_x[pj] - loc_x[pi], loc_y[pj] - loc_y[pi])
                        dist.append(d)
        if dist:
            dists_same_class += dist
            
    dists_same_class = np.sort(dists_same_class)

    if len(dists_same_class)<1:
        return None
    else:
        return dists_same_class

def plot_histogram(data, title, xlabel, ylabel, bins=50, log_y=True, show_pitch_twist={}, multi_crosshair=False):     
    from bokeh.plotting import ColumnDataSource, figure
    hist, edges = np.histogram(data, bins=bins)
    hist_linear = hist
    if log_y:
        hist = np.log10(1+hist)
    
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title=title, tools=tools, aspect_ratio=1)
    center = (edges[:-1]+edges[1:])/2
    source_data = dict(top=hist, left=edges[:-1], right=edges[1:], center=center, hist_linear=hist_linear)
    if show_pitch_twist:
        rise = show_pitch_twist["rise"]
        csyms = show_pitch_twist["csyms"]
        tmp = {}
        for csym in csyms:
            tmp[f"C{csym}"] = 360/(center*csym/rise)
        source_data.update(tmp)
    source = ColumnDataSource(data=source_data)
        
    hist_plot = fig.quad(top='top', bottom=0, left='left', right='right', fill_color="skyblue", source=source)
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = f"log10({ylabel})" if log_y else ylabel
    fig.title.align = "center"
    fig.title.text_font_size = "18px"
    fig.title.text_font_style = "normal"
    from bokeh.models.tools import HoverTool
    tooltips = [(xlabel, "@center{0} (@left{0}-@right{0})"), (ylabel, "@hist_linear")]
    if show_pitch_twist:
        for csym in show_pitch_twist["csyms"]:
            tooltips.append((f"Twist for C{csym} (°)", f"@C{csym}{{0.00}}"))
    hist_hover = HoverTool(renderers=[hist_plot], tooltips=tooltips)
    fig.add_tools(hist_hover)
    fig.hover[0].attachment="vertical"

    if multi_crosshair:
        from bokeh.models import Span, CustomJS
        n_crosshairs = 20
        crosshairs = []
        for i in range(2, n_crosshairs):
            crosshair = Span(location=0, dimension='height', line_dash='dashed')
            fig.add_layout(crosshair)
            crosshairs.append(crosshair)
        callback_code = "\n".join([f"crosshair_{i}.location = x*{i};" for i in range(2, n_crosshairs)])
        callback = CustomJS(args={f"crosshair_{i}": crosshairs[i-2] for i in range(2, n_crosshairs)}, code=f"""
            var x = cb_obj.x;
            {callback_code}
        """)
        fig.js_on_event('mousemove', callback)        
    return fig

def create_image_figure(image, dx, dy, title="", title_location="above", plot_width=None, plot_height=None, x_axis_label='x', y_axis_label='y', tooltips=None, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=None):
    from bokeh.plotting import ColumnDataSource, figure
    h, w = image.shape
    if aspect_ratio is None:
        if plot_width and plot_height:
            aspect_ratio = plot_width/plot_height
        else:
            aspect_ratio = w*dx/(h*dy)
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title_location=title_location, 
        frame_width=plot_width, frame_height=plot_height, 
        x_axis_label=x_axis_label, y_axis_label=y_axis_label,
        x_range=(-w//2*dx, (w//2-1)*dx), y_range=(-h//2*dy, (h//2-1)*dy), 
        tools=tools, aspect_ratio=aspect_ratio)
    fig.grid.visible = False
    if title:
        fig.title.text=title
        fig.title.align = "center"
        fig.title.text_font_size = "18px"
        fig.title.text_font_style = "normal"
    if not show_axis: fig.axis.visible = False
    if not show_toolbar: fig.toolbar_location = None

    source_data = ColumnDataSource(data=dict(image=[image], x=[-w//2*dx], y=[-h//2*dy], dw=[w*dx], dh=[h*dy]))
    from bokeh.models import LinearColorMapper
    color_mapper = LinearColorMapper(palette='Greys256')    # Greys256, Viridis256
    image = fig.image(source=source_data, image='image', color_mapper=color_mapper,
                x='x', y='y', dw='dw', dh='dh'
            )

    from bokeh.models.tools import HoverTool, CrosshairTool
    if not tooltips:
        tooltips = [("x", "$xÅ"), ('y', '$yÅ'), ('val', '@image')]
    image_hover = HoverTool(renderers=[image], tooltips=tooltips)
    fig.add_tools(image_hover)
    fig.hover[0].attachment="vertical"
    crosshair = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    if crosshair: 
        for ch in crosshair: ch.line_color = crosshair_color
    return fig

def get_pixel_size(data, attrs=["rlnMicrographOriginalPixelSize", "rlnMicrographPixelSize", "rlnImagePixelSize"], return_source=False):
    try:
        sources = [ data.attrs["optics"] ]
    except:
        sources = []
    sources += [data]
    for source in sources:
        for attr in attrs:
            if attr in source:
                if attr in ["rlnImageName", "rlnMicrographName"]:
                    import mrcfile, pathlib
                    folder = pathlib.Path(data["starFile"].iloc[0])
                    if folder.is_symlink(): folder = folder.readlink()
                    folder = folder.resolve().parent 
                    filename = source[attr].iloc[0].split("@")[-1]
                    filename = str((folder / "../.." / filename).resolve())
                    with mrcfile.open(filename, header_only=True) as mrc:
                        apix = float(mrc.voxel_size.x)
                else:
                    apix = float(source[attr].iloc[0])
                if return_source: return apix, attr
                else: return apix
    return None

def assign_segment_id(data, inter_segment_distance):
    assert "rlnHelicalTrackLengthAngst" in data
    tmp = data.loc[:, "rlnHelicalTrackLengthAngst"].astype(float) / inter_segment_distance
    err = (tmp - tmp.round()).abs()
    if np.sum(np.where(err > 0.1))>0:
        print(f"WARNING: it appears that the helical segments were extracted with different inter-segment distances")
    helical_segment_id = tmp.round().astype(int)
    return helical_segment_id

def estimate_inter_segment_distance(data):
    # data must have been sorted by micrograph, rlnHelicalTubeID, and rlnHelicalTrackLengthAngst
    helices = data.groupby(['rlnMicrographName', "rlnHelicalTubeID"], sort=False)

    import numpy as np
    dists_all = []
    for _, particles in helices:
        if len(particles)<2: continue
        dists = np.sort(particles["rlnHelicalTrackLengthAngst"].astype(float).values)
        dists = dists[1:] - dists[:-1]
        dists_all.append(dists)
    dists_all = np.hstack(dists_all)
    dist_seg = np.median(dists_all)   # Angstrom
    return dist_seg

def encode_numpy(img, hflip=False, vflip=False):
    if img.dtype != np.dtype('uint8'):
        vmin, vmax = img.min(), img.max()
        tmp = (255*(img-vmin)/(vmax-vmin)).astype(np.uint8)
    else:
        tmp = img
    if hflip:
        tmp = tmp[:, ::-1]
    if vflip:
        tmp = tmp[::-1, :]
    import io, base64
    from PIL import Image
    pil_img = Image.fromarray(tmp)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"

@st.cache_data(show_spinner=False)
def get_class2d_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_class2d_from_file(temp.name)

@st.cache_data(show_spinner=False)
def get_class2d_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    local_filename = download_file_from_url(url_final)
    if local_filename is None:
        st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    data = get_class2d_from_file(local_filename)
    return data

@st.cache_data(show_spinner=False)
def get_class2d_from_file(classFile):
    import mrcfile
    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, apix

@st.cache_data(show_spinner=False)
def get_class2d_params_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_class2d_params_from_file(temp.name)

@st.cache_data(show_spinner=False)
def get_class2d_params_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    local_filename = download_file_from_url(url_final)
    if local_filename is None:
        st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    data = get_class2d_params_from_file(local_filename)
    return data

@st.cache_data(show_spinner=False)
def get_class2d_params_from_file(params_file):
    if params_file.endswith(".star"):
        params = star_to_dataframe(params_file)
        return params

@st.cache_data(show_spinner=False)
def star_to_dataframe(starFile):
    import pandas as pd
    from gemmi import cif
    star = cif.read_file(starFile)
    if len(star) == 2:
        optics = pd.DataFrame()
        for item in star[0]:
            for tag in item.loop.tags:
                value = star[0].find_loop(tag)
                optics[tag.strip('_')] = np.array(value)
    else:
        optics = None

    data = pd.DataFrame()
    for item in star[-1]:
        for tag in item.loop.tags:
            value = star[-1].find_loop(tag)
            data[tag.strip('_')] = np.array(value)
    
    if optics is not None:
        data.attrs["optics"] = optics

    data["starFile"] = starFile
    return data

@st.cache_data(show_spinner=False)
def download_file_from_url(url):
    import requests
    try:
        filesize = get_file_size(url)
        local_filename = url.split('/')[-1]
        with st.spinner(f'Downloading {url} ({filesize/2**20:.1f} MB)'):
            with requests.get(url) as r:
                r.raise_for_status()  # Check for request success
                with open(local_filename, 'wb') as f:
                    f.write(r.content)
        return local_filename
    except requests.exceptions.RequestException as e:
        return None

@st.cache_data(show_spinner=False)
def get_file_size(url):
    import requests
    response = requests.head(url)
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])
        return file_size
    else:
        return None
    
def get_direct_url(url):
    import re
    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1")!=-1: return url
        elif url.find("dl=0")!=-1: return url.replace("dl=0", "dl=1")
        else: return url+"?dl=1"
    elif url.find("sharepoint.com")!=-1 and url.find("guestaccess.aspx")!=-1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64
        data_bytes64 = base64.b64encode(bytes(url, 'utf-8'))
        data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    else:
        return url

def import_with_auto_install(packages, scope=locals()):
    import importlib, site, pip
    if isinstance(packages, str): packages=[packages]
    for package in packages:
        if package.find(":")!=-1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = importlib.import_module(package_import_name)
        except:
            pip.main(['install', package_pip_name])
            importlib.reload(site)
            scope[package_import_name] = importlib.import_module(package_import_name)
import_with_auto_install("numpy pandas PIL:pillow bokeh:bokeh==2.4.3 requests mrcfile gemmi streamlit st_clickable_images".split())

if __name__ == "__main__":
    main()

