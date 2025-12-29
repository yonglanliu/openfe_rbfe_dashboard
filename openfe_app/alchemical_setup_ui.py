from __future__ import annotations

from pathlib import Path
import os
from typing import List, Optional, Tuple
from openfe_app.config import ProjectPaths

import streamlit as st
import pandas as pd
from rdkit import Chem
from openff.units import unit


# -------------------------
# Helpers
# -------------------------

def _rdkit_mols_from_sdf(sdf_path: Path) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    return [m for m in suppl if m is not None]


def _ensure_rdkit_names(mols: List[Chem.Mol]) -> None:
    for i, m in enumerate(mols):
        if not m.HasProp("_Name") or not m.GetProp("_Name").strip():
            m.SetProp("_Name", f"lig_{i}")


def _to_openfe_ligand_components(rdmols: List[Chem.Mol]):
    import openfe
    ligands = [openfe.SmallMoleculeComponent.from_rdkit(m) for m in rdmols]
    for m, c in zip(rdmols, ligands):
        try:
            c.name = m.GetProp("_Name")
        except Exception:
            pass
    return ligands


def _get_mapper(mapper_name: str):
    from openfe import setup
    if mapper_name == "LomapAtomMapper":
        return setup.LomapAtomMapper()
    if mapper_name == "KartografAtomMapper":
        return setup.KartografAtomMapper()
    raise ValueError(f"Unknown mapper: {mapper_name}")


def _get_ligand_network_planner(planner_name: str):
    import openfe
    planners = openfe.ligand_network_planning

    if planner_name == "minimal_spanning":
        return planners.generate_minimal_spanning_network, False
    if planner_name == "minimal_redundant":
        return planners.generate_minimal_redundant_network, False
    if planner_name == "radial":
        return planners.generate_radial_network, True

    raise ValueError(f"Unknown planner: {planner_name}")


def _default_lomap_scorer():
    try:
        from openfe import lomap_scorers
        return lomap_scorers.default_lomap_score
    except Exception:
        from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score
        return default_lomap_score


def _dump_transformation_to_json(t, out_path: Path) -> None:
    if hasattr(t, "to_json"):
        t.to_json(str(out_path))
        return
    if hasattr(t, "dump"):
        t.dump(str(out_path))
        return
    raise AttributeError("Transformation has no to_json() or dump() method.")


# -------------------------
#       Main Tab UI
# -------------------------

def alchemical_setup_tab(
    inputs_sdf: Path,
    protein_pdb: Path,
    planned_root: Path
):
    st.markdown(f"If you want to understand each parameter, go to "
                "**[OpenFE Tutorial](https://docs.openfree.energy/en/stable/cookbook/choose_protocol.html)** to learn.")
    st.subheader("Alchemical Setup (RBFE) → Dump Transformation JSONs")

    if not inputs_sdf.exists():
        st.warning("Upload ligands.sdf in Inputs first.")
        return
    if not protein_pdb.exists():
        st.warning("Upload protein.pdb in Inputs first.")
        return
    
    # Determine the out directory
    out_dir = (planned_root).resolve()
    st.caption(f"Planned output folder (created only on dump): `{out_dir}`")
   
    # -------------------------------------------
    # Campaign / output (NO folder creation here)
    # -------------------------------------------
    colA, colB = st.columns([2, 2])
    with colA:
        prefix_name = st.text_input(
            "Prefix Name",
            value="rbfe",
        )
                        
    button = st.button("Check JSON Files", type="primary")
    
    if button:
        if any(Path(out_dir).glob("*.json")):
            prefix_name_exist = set()
            for p in os.listdir(out_dir):
                name = p.split("_")
                prefix_name_exist.add(name[0])
            if prefix_name in prefix_name_exist:
                st.info(f"""JSON file with \"{prefix_name}\" exist.
                        If you don't want to overwrite them, 
                        you must choose another prefix name""")
            else:
                st.info("""This name is safe. Feel free to use it!""")
        else:
            st.info("""There isn't any JSON files. Free free to use this name!""")

    st.divider()

    st.markdown('<div align="center" style="font-size:36px; font-weight:bold;">System Setup</div>', 
                unsafe_allow_html=True)

    # -------------------------
    #      Network Planner
    # -------------------------
    st.markdown("### Network planner")
    c1, c2 = st.columns([2, 2])

    with c1:
        mapper_name = st.selectbox(
            "Atom mapper",
            ["LomapAtomMapper", "KartografAtomMapper"],
            index=0,
        )

    with c2:
        ln_planner = st.selectbox(
            "Ligand network planner",
            ["minimal_spanning", "minimal_redundant", "radial"],
            index=0
        )

    radial_center_name = None
    if ln_planner == "radial":
        mols_tmp = _rdkit_mols_from_sdf(inputs_sdf)
        _ensure_rdkit_names(mols_tmp)
        names_tmp = [m.GetProp("_Name") for m in mols_tmp]
        radial_center_name = st.selectbox(
            "Central ligand (radial network)",
            options=names_tmp,
            index=0,
        )

    st.divider()

    # ----------------------------------
    #    platform/parellel Setting
    # ----------------------------------
    
    c1, _, c2 = st.columns([2, 0.5, 2])
    with c1:
        st.markdown("### Platform Setting")
        platform = st.selectbox(
            "Choose platform",
            ["cuda", "opencl", "cpu"],
            index=0
        )
    with c2:
        st.markdown("### Parellel Schedule")
        protocol_repeats = st.number_input(
            "Protocol repeats",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

    st.divider()

    # -------------------------------------
    #  Nonbonded/lambda/integrator Setting
    # -------------------------------------

    c1, _, c2, _, c3, = st.columns([2, 0.3, 2, 0.3, 2])

    with c1:
        st.markdown("### Nonbonded Setting")
        cc1, cc2 = st.columns([2, 2])
        with cc1:
            nonb_method = st.text_input(
                "Nonbonded Method",
                value = "PME",
            )
        with cc2:
            nonb_cutoff = float(st.text_input(
                "LJ cutoff (nm)",
                value = 0.9,
            ))

    with c2:
        st.markdown("### Lambda Schedule")
        n_lambda_windows = st.number_input(
            f"Number of $\lambda$ windows",
            min_value=1,
            value = 11,
            step = 1
        )

    with c3:
        st.markdown("### Integrator Setting")
        time_step = st.number_input(
            "Integration Timestep",
            min_value=1,
            max_value=8,
            value = 4,
            step = 1
        )

    st.divider()
    
    # -------------------------
    #    Thermo Setting
    # -------------------------
    st.markdown("### Thermo Setting")
    c1, _, c2= st.columns([2, 0.5, 2])
    with c1:
        temp = float(st.text_input(
            "Temperature (Kelvin)",
            value=300,
        ))
    with c2:
        pressure = int(st.text_input(
            "Barostat Pressure (bar)",
            value = 1,
        ))
    st.divider()

    # -------------------------
    #    Solvation Setting
    # -------------------------
    st.markdown("### Solvation Setting")
    c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 2, 2, 2])
    with c1:
        sol_model = st.selectbox(
            "Solvent Model",
            ["tip3p", "tip3p-fb", "spc", "spce", "tip4p", "tip4p-ew", "tip4p-fb"],
            index=0,
        )
    with c2:
        sol_padding = float(st.text_input(
            "Solvent Padding (nm)",
            value = 1.5,
        ))
    with c3:
        neutralize = st.selectbox(
            "Add Counterions",
            ["True", "False"],
            index=0
        )
    with c4:
        ion_pos = st.selectbox(
            "Positive Ions",
            ["Na"],
            index=0
        )
    with c5:
        ion_neg = st.selectbox(
            "Negative Ions",
            ["Cl"],
            index=0
        )
    with c6:
        ion_con = float(st.text_input(
            "Ion Concentration (mol)",
            value = 0.15
        ))
    st.divider()

    # -------------------------
    #    Simulation Setting
    # -------------------------
    st.markdown("### Simulation Setting")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        
        mini_step = int(st.text_input(
            "Minimization Steps",
            value=5000,
        ))
    with c2: 
        time_per_iteration = float(st.text_input(
            "Sampling Time Interval (ps)",
            value = 1.0
        ))
    with c3:
        equil_sim_time = float(st.text_input(
            "Equilibrium Time (ns)",
            value=1
        ))
    with c4:
        prod_sim_time = float(st.text_input(
            "Production Time (ns)", 
            value = 5
        ))

    st.divider()

    # -------------------------
    #    Output Setting
    # -------------------------
    st.markdown("### Output Setting")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        output_file_name = st.text_input(
            "Output File Name",
            value = "simulation.nc"
        )
    with c2:
        output_structure = st.text_input(
            "Ouput Structure Name",
            value="hybrid_system.pdb",
        )
    with c3:
        chk_interval = float(st.text_input(
            "Checkpoing Interval (ns)",
            value=1
        ))
    with c4:
        pos_freq = float(st.text_input(
            "Position Write Frequency (ps)",
            value=100
        ))
    st.divider()

    # -------------------------
    #    Main action button
    # -------------------------
    left, center, right = st.columns([1, 1, 1])

    with center:
        button = st.button(
            "Create RBFE alchemical system + dump JSON",
            type="primary",
        )

    if button:
        # Create output folder ONLY here
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load ligands ----
        mols = _rdkit_mols_from_sdf(inputs_sdf)
        if not mols:
            st.error("No valid ligands found in ligands.sdf")
            return
        _ensure_rdkit_names(mols)

        ligands = _to_openfe_ligand_components(mols)

        import openfe
        protein = openfe.ProteinComponent.from_pdb_file(str(protein_pdb))
        solvent = openfe.SolventComponent(positive_ion=ion_pos, negative_ion=ion_neg,
                           neutralize=neutralize, ion_concentration=ion_con * unit.molar)

        mapper = _get_mapper(mapper_name)
        planner_fn, needs_center = _get_ligand_network_planner(ln_planner)
        scorer = _default_lomap_scorer()
    
        #protocol = _build_protocol(platform=None, 
        #                           protocol_repeats=int(protocol_repeats)).default_settings()
        from openfe.protocols.openmm_rfe import (
            RelativeHybridTopologyProtocol,
            RelativeHybridTopologyProtocolSettings,
        )
        settings = RelativeHybridTopologyProtocol.default_settings()
        #st.write(settings)

        # ---------------------------------------
        #          Change parameters
        # ---------------------------------------

        # Nonbonded setting
        settings.forcefield_settings.nonbonded_method = nonb_method
        settings.forcefield_settings.nonbonded_cutoff = nonb_cutoff * unit.nanometer
        # Lambda setting
        settings.lambda_settings.lambda_windows = n_lambda_windows
        # Integrator setting
        settings.integrator_settings.timestep = time_step * unit.femtosecond
        # Thermo setting
        settings.thermo_settings.temperature=temp * unit.kelvin
        settings.thermo_settings.pressure = pressure * unit.bar
        # Solvation setting
        settings.solvation_settings.solvent_model = sol_model
        settings.solvation_settings.solvent_padding = sol_padding * unit.nanometer
        # Simulation setting
        settings.simulation_settings.minimization_steps = mini_step
        settings.simulation_settings.equilibration_length = equil_sim_time * unit.nanosecond
        settings.simulation_settings.production_length = prod_sim_time * unit.nanosecond
        settings.simulation_settings.time_per_iteration=time_per_iteration * unit.ps
        settings.simulation_settings.n_replicas = n_lambda_windows
        # Output setting
        settings.output_settings.output_filename = output_file_name
        settings.output_settings.output_structure = output_structure
        settings.output_settings.checkpoint_interval = chk_interval * unit.nanosecond
        settings.output_settings.positions_write_frequency = pos_freq * unit.picosecond

        settings.protocol_repeats = protocol_repeats
        settings.engine_settings.compute_platform = platform

        #st.write(settings)

        protocol = RelativeHybridTopologyProtocol(settings=settings)
        #-----------------------------------------

        from openfe.setup.alchemical_network_planner import RBFEAlchemicalNetworkPlanner

        rbfe_planner = RBFEAlchemicalNetworkPlanner(
            name=prefix_name,
            mappers=[mapper],
            mapping_scorer=scorer,
            ligand_network_planner=planner_fn,
            protocol=protocol,
        )

        if needs_center:
            name_to_comp = {c.name: c for c in ligands}
            if radial_center_name is None:
                st.error("Radial network requires selecting a central ligand.")
                return
            center_comp = name_to_comp[radial_center_name]
            alchemical_network = rbfe_planner(
                ligands=ligands,
                solvent=solvent,
                protein=protein,
                ligand_network_planner_kwargs={"central_ligand": center_comp},
            )
        else:
            alchemical_network = rbfe_planner(
                ligands=ligands,
                solvent=solvent,
                protein=protein,
            )

        transformations = list(alchemical_network.edges)
        if not transformations:
            st.error("No transformations created.")
            return

        # ---- Dump JSONs ----
        written = []
        for i, t in enumerate(transformations, start=1):
            a = getattr(t.stateA, "name", "A")
            b = getattr(t.stateB, "name", "B")
            name = a + "_" + b

            fname = f"{prefix_name}_{name}".replace(" ", "_").replace("/", "_")
            #st.write(fname)
            json_path = out_dir / f"{fname}.json"

            _dump_transformation_to_json(t, json_path)
            written.append(json_path)

        st.success(f"Wrote {len(written)} JSON files to {out_dir}")

    # -------------------------
    # Show current JSONs
    # (only if folder exists and contains JSONs)
    # -------------------------
    st.divider()
    st.subheader("Current dumped transformation JSONs")

    if not out_dir.exists():
        st.info("No campaign folder created yet. Click the button above to dump JSON files.")
        return

    tx = sorted(out_dir.glob("*.json"))
    if not tx:
        st.info("No JSONs in this campaign yet.")
        return

    df = pd.DataFrame([{"json": p.name, "path": str(p)} for p in tx])
    st.dataframe(df, use_container_width=True)
