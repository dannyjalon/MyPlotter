import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import AutoMinorLocator

# --- Global Plotting Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4
})

# --- Standard Colors ---
STANDARD_COLORS = {
    "Blue": "#0072BD",
    "Orange": "#D95319",
    "Yellow": "#EDB120",
    "Purple": "#7E2F8E",
    "Green": "#77AC30",
    "Cyan": "#4DBEEE",
    "Maroon": "#A2142F",
    "Black": "#000000",
    "Red": "#FF0000",
    "Dark Blue": "#0000FF",
    "Dark Green": "#008000"
}


def calculate_trendline(x, y, type_name):
    """Calculates trendline x, y, and equation string including R-squared."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2: return None, None, ""

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    try:
        if type_name == "Linear":
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)
            eq = f"y = {z[0]:.2f}x + {z[1]:.2f}"
        elif type_name == "Exponential":
            pos_mask = y > 0
            x_f, y_f = x[pos_mask], y[pos_mask]
            if len(x_f) < 2: return None, None, ""
            z = np.polyfit(x_f, np.log(y_f), 1)
            A = np.exp(z[1])
            B = z[0]
            y_pred = A * np.exp(B * x_f)
            eq = f"y = {A:.2f}e^({B:.2f}x)"
            x, y = x_f, y_f
        elif type_name == "Logarithmic":
            pos_mask = x > 0
            x_f, y_f = x[pos_mask], y[pos_mask]
            if len(x_f) < 2: return None, None, ""
            z = np.polyfit(np.log(x_f), y_f, 1)
            A = z[0]
            B = z[1]
            y_pred = A * np.log(x_f) + B
            eq = f"y = {A:.2f}ln(x) + {B:.2f}"
            x, y = x_f, y_f
        elif type_name == "Power":
            pos_mask = (x > 0) & (y > 0)
            x_f, y_f = x[pos_mask], y[pos_mask]
            if len(x_f) < 2: return None, None, ""
            z = np.polyfit(np.log(x_f), np.log(y_f), 1)
            A = np.exp(z[1])
            B = z[0]
            y_pred = A * (x_f ** B)
            eq = f"y = {A:.2f}x^({B:.2f})"
            x, y = x_f, y_f
        elif type_name == "Polynomial (2nd Order)":
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            y_pred = p(x)
            eq = f"y = {z[0]:.2f}x² + {z[1]:.2f}x + {z[2]:.2f}"
        else:
            return None, None, ""

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return x, y_pred, f"{eq}\n($R^2={r2:.3f}$)"

    except Exception:
        return None, None, "Fit Error"


def create_advanced_plot(df, df_err, x_col, series_configs, plot_settings):
    fig, ax = plt.subplots(figsize=(12, 10))

    for y_col, config in series_configs.items():
        # --- Data Prep ---
        temp_data = {'x': df[x_col], 'y': df[y_col]}
        has_error = False

        if df_err is not None and config['error_col'] != "None" and config['error_col'] in df_err.columns:
            min_len = min(len(df), len(df_err))
            temp_data['x'] = df[x_col].iloc[:min_len]
            temp_data['y'] = df[y_col].iloc[:min_len]
            temp_data['err'] = df_err[config['error_col']].iloc[:min_len]
            has_error = True

        sub_df = pd.DataFrame(temp_data).dropna()
        if sub_df.empty: continue

        x_raw = sub_df['x'].values
        y_raw = sub_df['y'].values
        y_err = sub_df['err'].values if has_error else None

        # --- Styles ---
        label_str = config['custom_label']
        ls = config['linestyle']
        mk = config['marker']
        ms = config['marker_size']
        color = config['color']
        capsize = config['capsize']
        err_style = config['error_style']  # Bar or Sleeve
        sleeve_alpha = config['sleeve_alpha']

        if config['marker_fill'] == 'Hollow':
            mfc = 'white'
            mec = color
        else:
            mfc = color
            mec = color

        # --- PLOTTING LOGIC ---

        # 1. Plot Error (Sleeve or Bar) - HIDDEN FROM LEGEND
        if has_error:
            if err_style == "Sleeve":
                # SLEEVE LOGIC
                y_lower = y_raw - y_err
                y_upper = y_raw + y_err

                if config['smoothing']:
                    # Smooth the boundaries too!
                    sort_idx = np.argsort(x_raw)
                    x_sorted = x_raw[sort_idx]
                    y_l_sorted = y_lower[sort_idx]
                    y_u_sorted = y_upper[sort_idx]

                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
                    try:
                        # We need to spline lower and upper bounds
                        # Note: assuming monotonic X for spline, or use unique check
                        unique_x, indices = np.unique(x_sorted, return_index=True)
                        if len(unique_x) > 3:
                            spl_l = make_interp_spline(unique_x, y_l_sorted[indices], k=3)
                            spl_u = make_interp_spline(unique_x, y_u_sorted[indices], k=3)
                            ax.fill_between(x_smooth, spl_l(x_smooth), spl_u(x_smooth), color=color, alpha=sleeve_alpha,
                                            linewidth=0)
                        else:
                            ax.fill_between(x_raw, y_lower, y_upper, color=color, alpha=sleeve_alpha, linewidth=0)
                    except:
                        ax.fill_between(x_raw, y_lower, y_upper, color=color, alpha=sleeve_alpha, linewidth=0)
                else:
                    # No smoothing sleeve
                    ax.fill_between(x_raw, y_lower, y_upper, color=color, alpha=sleeve_alpha, linewidth=0)

            else:
                # BAR LOGIC (Standard)
                ax.errorbar(x_raw, y_raw, yerr=y_err,
                            fmt='none',
                            ecolor=color, elinewidth=1.5, capsize=capsize,
                            label=None)

        # 2. Draw Main Data (Smooth or Linear) - HIDDEN FROM LEGEND
        if config['smoothing']:
            sort_idx = np.argsort(x_raw)
            x_sorted, y_sorted = x_raw[sort_idx], y_raw[sort_idx]
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            try:
                unique_x, indices = np.unique(x_sorted, return_index=True)
                unique_y = y_sorted[indices]
                if len(unique_x) > 3:
                    spl = make_interp_spline(unique_x, unique_y, k=3)
                    plot_x, plot_y = x_smooth, spl(x_smooth)
                else:
                    plot_x, plot_y = x_sorted, y_sorted
            except:
                plot_x, plot_y = x_sorted, y_sorted

            if ls != "None":
                ax.plot(plot_x, plot_y, color=color, linestyle=ls, linewidth=2.0, label=None)

            if mk != "None":
                ax.plot(x_raw, y_raw, linestyle='None', marker=mk, markersize=ms,
                        markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2, label=None)

        else:
            # NO SMOOTHING
            final_ls = ls if ls != "None" else "none"
            final_mk = mk if mk != "None" else "none"
            ax.plot(x_raw, y_raw, color=color, linestyle=final_ls, linewidth=2.0,
                    marker=final_mk, markersize=ms,
                    markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2,
                    label=None)

        # 3. LEGEND GHOST HANDLE
        final_ls_leg = ls if ls != "None" else "None"
        final_mk_leg = mk if mk != "None" else "None"
        ax.plot([], [], color=color, linestyle=final_ls_leg, linewidth=2.0,
                marker=final_mk_leg, markersize=ms,
                markeredgecolor=mec, markerfacecolor=mfc, markeredgewidth=2,
                label=label_str)

        # 4. Trendlines
        if config['trendline'] != "None":
            tx, ty, t_label = calculate_trendline(x_raw, y_raw, config['trendline'])
            if tx is not None:
                final_label = f"{t_label}" if config['show_eq'] else None
                ax.plot(tx, ty, color=color, linestyle=':', linewidth=1.5, alpha=0.8, label=final_label)

    # --- Axes & Legends ---
    if plot_settings['title']:
        ax.set_title(plot_settings['title'], fontsize=plot_settings['fs_title'], fontname="Times New Roman", pad=15)

    ax.set_xlabel(plot_settings['x_label'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")
    ax.set_ylabel(plot_settings['y_label'], fontsize=plot_settings['fs_label'], fontname="Times New Roman")
    ax.tick_params(axis='both', which='major', labelsize=plot_settings['fs_ticks'])

    if plot_settings['minor_ticks_x'] > 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_x'] + 1))
    if plot_settings['minor_ticks_y'] > 0:
        ax.yaxis.set_minor_locator(AutoMinorLocator(plot_settings['minor_ticks_y'] + 1))
    if plot_settings['minor_ticks_x'] > 0 or plot_settings['minor_ticks_y'] > 0:
        ax.tick_params(which='minor', direction='in', top=True, right=True)

    if plot_settings['x_lim']: ax.set_xlim(plot_settings['x_lim'])
    if plot_settings['y_lim']: ax.set_ylim(plot_settings['y_lim'])

    if plot_settings['legend_loc'] != "None":
        ax.legend(loc=plot_settings['legend_loc'],
                  fontsize=plot_settings['fs_legend'],
                  frameon=plot_settings['legend_frame'],
                  fancybox=False, edgecolor='black')

    return fig


# --- Streamlit UI ---
st.set_page_config(page_title="Pro Plotter V12", layout="wide")
st.title("📈 Academic Plotter V12")

with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload Data", type=["xlsx", "csv"])
    uploaded_err = st.file_uploader("Upload Errors", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df_err = None
    if uploaded_err:
        if uploaded_err.name.endswith('csv'):
            df_err = pd.read_csv(uploaded_err)
        else:
            df_err = pd.read_excel(uploaded_err)

    with st.expander("🔎 Data Preview", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Data File**")
            st.dataframe(df, use_container_width=True, height=150)
        with c2:
            st.write("**Error File**")
            if df_err is not None:
                st.dataframe(df_err, use_container_width=True, height=150)
            else:
                st.info("No error file uploaded")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Data Setup")
        cols = df.columns.tolist()
        x_axis = st.selectbox("X Axis", cols)
        y_axes = st.multiselect("Y Axis (Curves)", [c for c in cols if c != x_axis])

        err_cols_list = ["None"]
        if df_err is not None:
            err_cols_list += df_err.columns.tolist()

        series_configs = {}
        if y_axes:
            st.markdown("---")
            st.subheader("Curve Styling")
            for i, y_col in enumerate(y_axes):
                with st.expander(f"🖌️ {y_col}", expanded=(i == 0)):
                    custom_label = st.text_input("Label", value=y_col, key=f"lbl_{y_col}")

                    # Error Matching
                    err_def_idx = 0
                    if df_err is not None:
                        if y_col in df_err.columns:
                            err_def_idx = err_cols_list.index(y_col)
                        else:
                            try:
                                col_idx = df.columns.get_loc(y_col)
                                if col_idx < len(df_err.columns):
                                    match_col = df_err.columns[col_idx]
                                    err_def_idx = err_cols_list.index(match_col)
                            except:
                                pass

                    error_col = st.selectbox("Error Data", err_cols_list, index=err_def_idx, key=f"ec_{y_col}")

                    err_style = "Bar"
                    sleeve_alpha = 0.2
                    capsize = 4

                    if error_col != "None":
                        st.caption(f"✅ Linked to: **{error_col}**")
                        # New: Selector for Style
                        e1, e2 = st.columns(2)
                        err_style = e1.selectbox("Err Style", ["Bar", "Sleeve"], key=f"es_{y_col}")

                        if err_style == "Bar":
                            capsize = e2.slider("Cap Size", 0, 10, 4, key=f"cs_{y_col}")
                        else:
                            sleeve_alpha = e2.slider("Opacity", 0.0, 1.0, 0.2, key=f"sa_{y_col}")
                    else:
                        st.caption("❌ No errors linked")

                    # Colors
                    c1, c2 = st.columns([1, 1])
                    preset_name = c1.selectbox("Color Preset", list(STANDARD_COLORS.keys()),
                                               index=i % len(STANDARD_COLORS), key=f"ps_{y_col}")
                    final_color = c2.color_picker("Custom", STANDARD_COLORS[preset_name], key=f"cp_{y_col}")

                    l1, l2 = st.columns(2)
                    linestyle = l1.selectbox("Line", ["-", "--", "-.", ":", "None"], key=f"ls_{y_col}")
                    smoothing = l2.checkbox("Smooth", value=False, key=f"sm_{y_col}")

                    m1, m2 = st.columns(2)
                    marker_shape = m1.selectbox("Marker", ["None", "Circle", "Square", "Triangle", "Diamond"],
                                                key=f"mk_{y_col}")
                    marker_fill = m2.selectbox("Fill", ["Solid", "Hollow"], key=f"mf_{y_col}")
                    ms = st.slider("Size", 1, 20, 8, key=f"ms_{y_col}")

                    series_configs[y_col] = {
                        "custom_label": custom_label,
                        "error_col": error_col,
                        "error_style": err_style,
                        "sleeve_alpha": sleeve_alpha,
                        "capsize": capsize,
                        "color": final_color,
                        "linestyle": linestyle,
                        "smoothing": smoothing,
                        "marker": {"None": "None", "Circle": "o", "Square": "s", "Triangle": "^", "Diamond": "D"}[
                            marker_shape],
                        "marker_fill": marker_fill,
                        "marker_size": ms,
                        "trendline": st.selectbox("Trendline", ["None", "Linear", "Exponential", "Logarithmic", "Power",
                                                                "Polynomial"], key=f"tr_{y_col}"),
                        "show_eq": st.checkbox("Show R²", value=True, key=f"seq_{y_col}")
                    }

    with col_right:
        st.subheader("Settings")
        with st.expander("Axes & Legend", expanded=True):
            t1, t2 = st.columns([3, 1])
            plot_title = t1.text_input("Title", "")
            fs_title = t2.number_input("Title Size", 10, 50, 24)

            l1, l2, l3 = st.columns([2, 2, 1])
            x_lbl = l1.text_input("X Label", r"$\beta$")
            y_lbl = l2.text_input("Y Label", r"$b_{opt}$")
            fs_lbl = l3.number_input("Lbl Size", 10, 50, 36)

            tk1, tk2, tk3 = st.columns(3)
            fs_tick = tk1.number_input("Tick Size", 10, 50, 32)
            fs_leg = tk2.number_input("Leg Size", 10, 50, 24)
            leg_pos = tk3.selectbox("Leg Pos", ["best", "upper right", "upper left", "lower right", "None"])
            leg_frame = st.checkbox("Frame", value=False)

            mt1, mt2 = st.columns(2)
            min_x = mt1.number_input("Min X Ticks", 0, 10, 0)
            min_y = mt2.number_input("Min Y Ticks", 0, 10, 0)


            def parse(s): return [float(x) for x in s.split(',')] if ',' in s else None


            x_lim = parse(st.text_input("X Lim", ""))
            y_lim = parse(st.text_input("Y Lim", ""))

            plot_settings = {
                "title": plot_title, "fs_title": fs_title,
                "x_label": x_lbl, "y_label": y_lbl, "fs_label": fs_lbl,
                "fs_ticks": fs_tick, "fs_legend": fs_leg,
                "x_lim": x_lim, "y_lim": y_lim, "legend_loc": leg_pos,
                "legend_frame": leg_frame,
                "minor_ticks_x": min_x, "minor_ticks_y": min_y
            }

        if y_axes:
            fig = create_advanced_plot(df, df_err, x_axis, series_configs, plot_settings)

            st.markdown("### Preview")
            c1, c2, c3 = st.columns([1, 6, 1])
            with c2:
                st.pyplot(fig, use_container_width=True)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("Download PNG", buf.getvalue(), "academic_plot_v12.png", "image/png")
        else:
            st.info("Select data to generate plot.")
else:
    st.info("Please upload a file to start.")