import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_csv(path, label_col="Cảnh báo học tập (0/1)"):
    """Load CSV and return dataframe."""
    # Try reading csv using pandas. We'll sanitize numeric strings containing commas.
    df = pd.read_csv(path)
    # sanitize string numeric values that use comma as decimal point (e.g., '7,46')
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            # Normalize decimals: replace comma decimals with dot. Remove thousands separators or spaces.
            new = s.str.replace(',', '.', regex=False).str.replace(' ', '')
            converted = pd.to_numeric(new, errors='coerce')
            # If many values converted successfully, use numeric column
            if converted.notna().sum() > 0:
                df[c] = converted
    return df


def prepare_sequences(df, feature_prefixes=["Điểm TB HK"], label_col="Cảnh báo học tập (0/1)"):
    """Searches for semester columns with pattern `feature_prefixes + <n>` and builds sequences.

    Returns X (n_samples, timesteps, features) and y (n_samples,)
    """
    # Detect semester columns for each prefix
    semester_cols = {}
    for pref in feature_prefixes:
        cols = sorted([c for c in df.columns if pref in c])
        if cols:
            semester_cols[pref] = cols

    # Basic features: numeric features excluding label and semester columns
    exclude_cols = [label_col]
    for pref_cols in semester_cols.values():
        exclude_cols += pref_cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # include numeric_cols that are not semester columns and not label
    base_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Include categorical columns as label-encoded features
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in exclude_cols]

    # Encode categorical columns
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = df[c].fillna('missing')
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    # Make sequence arrays
    # For now, we only use one prefix for sequences (e.g., "Điểm TB HK")
    # If multiple, concatenate along features dimension.
    if semester_cols:
        prefix_used = list(semester_cols.keys())[0]
        seq_cols = semester_cols[prefix_used]
        # Create sequences
        seq_data = df[seq_cols].fillna(0).values.astype(float)
        # shape (n_samples, timesteps) -> expand features dim to 1
        X_seq = seq_data.reshape((len(df), len(seq_cols), 1))
    else:
        # If no semester columns, create a sequence of length 1 from base features
        X_seq = None

    # Base features
    base_features = df[base_cols + cat_cols].fillna(0).values.astype(float)

    # Standard scale base features
    scaler = StandardScaler()
    base_features = scaler.fit_transform(base_features)

    if X_seq is not None:
        # concatenate base features to every timestep
        n, t, f = X_seq.shape
        base_expanded = np.repeat(base_features[:, np.newaxis, :], t, axis=1)
        X = np.concatenate([X_seq, base_expanded], axis=2)
    else:
        # create sequence length 1
        X = base_features[:, np.newaxis, :]

    # label
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not found in data")
    y = df[label_col].astype(int).values

    return X, y, scaler, encoders, base_cols, cat_cols, seq_cols


def prepare_sequences_for_prediction(df, scaler=None, encoders=None, metadata=None, feature_prefixes=["Điểm TB HK"]):
    """Prepare X for prediction; does not require label column.

    If scaler/encoders are provided (from training), they will be used to encode categorical vars and scale base features.
    """
    # Detect semester columns for each prefix
    semester_cols = {}
    for pref in feature_prefixes:
        cols = sorted([c for c in df.columns if pref in c])
        if cols:
            semester_cols[pref] = cols

    exclude_cols = []
    for pref_cols in semester_cols.values():
        exclude_cols += pref_cols

    # Use base_cols from metadata, if provided; otherwise infer numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if metadata and 'base_cols' in metadata:
        base_cols = metadata['base_cols']
    else:
        base_cols = [c for c in numeric_cols if c not in exclude_cols]

    if metadata and 'cat_cols' in metadata:
        cat_cols = metadata['cat_cols']
    else:
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in exclude_cols]

    # Encode categorical columns using provided encoders or new ones
    if encoders is None:
        encoders = {}
        for c in cat_cols:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[c] = df[c].fillna('missing')
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
    else:
        for c in cat_cols:
            le = encoders.get(c)
            if le is not None:
                df[c] = df[c].fillna('missing')
                df[c] = le.transform(df[c].astype(str))
            else:
                # if encoder missing, fallback to string hash
                df[c] = df[c].fillna('missing').astype(str).apply(lambda x: hash(x) % 1000)

    base_features = df[base_cols + cat_cols].fillna(0).values.astype(float)

    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        base_features = scaler.fit_transform(base_features)
    else:
        # Align features count with scaler's expected input dim
        try:
            expected = getattr(scaler, 'n_features_in_', None)
            if expected is not None:
                cur = base_features.shape[1]
                if cur > expected:
                    # If metadata available, try to reduce columns to metadata order
                    if metadata and 'base_cols' in metadata:
                        # pick only columns in base_cols and cat_cols order
                        cols = metadata['base_cols'] + metadata.get('cat_cols', [])
                        # compute intersection with current df columns
                        cols = [c for c in cols if c in df.columns]
                        base_features = df[cols].fillna(0).values.astype(float)
                        cur = base_features.shape[1]
                        if cur > expected:
                            base_features = base_features[:, :expected]
                    else:
                        base_features = base_features[:, :expected]
                elif cur < expected:
                    # pad zeros
                    pad = np.zeros((base_features.shape[0], expected - cur))
                    base_features = np.hstack([base_features, pad])
        except Exception:
            pass
        base_features = scaler.transform(base_features)

    # If metadata provides seq_cols order, use it
    if metadata and 'seq_cols' in metadata:
        seq_cols = metadata['seq_cols']
    else:
        seq_cols = None
        if semester_cols:
            prefix_used = list(semester_cols.keys())[0]
            seq_cols = semester_cols[prefix_used]

    if seq_cols is None:
        if semester_cols:
            prefix_used = list(semester_cols.keys())[0]
            seq_cols = semester_cols[prefix_used]
    if seq_cols:
        seq_data = df[seq_cols].fillna(0).values.astype(float)
        X_seq = seq_data.reshape((len(df), len(seq_cols), 1))
        n, t, f = X_seq.shape
        base_expanded = np.repeat(base_features[:, np.newaxis, :], t, axis=1)
        X = np.concatenate([X_seq, base_expanded], axis=2)
    else:
        X = base_features[:, np.newaxis, :]

    return X, scaler, encoders


def pivot_long_to_wide(df, student_id_col, semester_col, value_cols, max_timesteps=None, sort_col=None, agg='mean'):
    """Pivot a long-format dataset (student_id, semester, feature) into wide per-student sequences.

    df: pd.DataFrame
    student_id_col: column name that identifies students
    semester_col: column name identifying semester or time order
    value_cols: list of feature column names to pivot across semesters (e.g., list of scores)
    max_timesteps: optional int to limit time steps (pad or crop to this length)
    sort_col: if provided, will sort rows by this before pivoting (e.g., semester value coerced to int)
    Returns a wide DataFrame where columns are <feature>_t1, <feature>_t2, ... up to max_timesteps or the max found.
    """
    # Ensure semester is sorted
    if sort_col:
        df = df.sort_values([student_id_col, sort_col])
    else:
        df = df.sort_values([student_id_col, semester_col])

    # Create time index per student
    df['_timeidx'] = df.groupby(student_id_col).cumcount() + 1
    # If max_timesteps, filter or pad later
    if max_timesteps:
        df = df[df['_timeidx'] <= max_timesteps]

    # For each value column, pivot
    pieces = []
    for col in value_cols:
        pivot = df.pivot(index=student_id_col, columns='_timeidx', values=col)
        pivot.columns = [f"{col}_t{c}" for c in pivot.columns]
        pieces.append(pivot)

    wide = pd.concat(pieces, axis=1)
    wide.index.name = student_id_col
    wide = wide.reset_index()

    # If max_timesteps, ensure columns are present for all timesteps (fill with NaN)
    if max_timesteps:
        expected_cols = []
        for col in value_cols:
            for i in range(1, max_timesteps + 1):
                expected_cols.append(f"{col}_t{i}")
        for c in expected_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        wide = wide[[student_id_col] + expected_cols]

    return wide


def prepare_sequences_from_long(df, student_id_col, semester_col, feature_cols, label_col=None, max_timesteps=None, sort_col=None):
    """Prepare sequences from long-format table and return X, y, scaler, encoders, base_cols, cat_cols, seq_cols.
    """
    wide = pivot_long_to_wide(df, student_id_col, semester_col, value_cols=feature_cols, max_timesteps=max_timesteps, sort_col=sort_col)
    # Merge back with labels and base features (take first occurrence per student)
    base_cols = [c for c in df.columns if c not in ([student_id_col, semester_col] + feature_cols + [label_col])]
    # Aggregate first row per student as base features
    first = df.sort_values([student_id_col, semester_col]).groupby(student_id_col).first().reset_index()[[student_id_col] + base_cols + ([label_col] if label_col else [])]
    merged = first.merge(wide, on=student_id_col, how='left')
    # Use prepare_sequences to process merged wide-frame by detecting the sequence columns via prefixes
    X, y, scaler, encoders, base_cols_out, cat_cols, seq_cols = prepare_sequences(merged, feature_prefixes=[f + '_t' for f in feature_cols], label_col=label_col)
    return X, y, scaler, encoders, base_cols_out, cat_cols, seq_cols
