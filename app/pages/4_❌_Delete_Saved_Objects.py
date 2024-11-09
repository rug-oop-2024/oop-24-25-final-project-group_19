import streamlit as st
from app.core.system import AutoMLSystem


st.set_page_config(page_title="Delete", page_icon="❌")

class Delete:
    def _delete(self):
        automl = AutoMLSystem.get_instance()
        artifact_type = st.selectbox("What would you like to delete?", ("Pipeline", "Dataset"))
        
        if artifact_type:
            # List artifacts based on selected type
            artifacts = automl.registry.list(type="pipeline" if artifact_type == "Pipeline" else "dataset")
            artifact_names = {artifact.name: artifact for artifact in artifacts}
            
            # Select multiple artifacts to delete
            selected_artifacts = st.selectbox(
                f"Select {artifact_type} to delete:", list(artifact_names.keys())
            )
            
            # Confirm deletion for each selected artifact
            for name in selected_artifacts:
                artifact = artifact_names.get(name)
                st.write(f"Are you sure you want to delete {name} (version {artifact.version})?")
                
                if st.button(f"Delete {name}", key=f"delete_{name}"):
                    # Retrieve ID and delete artifact
                    artifact_id = automl.registry.get_id(name, artifact.version)
                    if artifact_id:
                        automl.registry.delete(artifact_id)
                        st.success(f"{artifact_type} '{name}' deleted successfully!")
                    else:
                        st.error(f"{artifact_type} '{name}' not found.")
    
    def run(self):
        self._delete()

delete = Delete()
delete.run()