import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api";

type View = "dashboard" | "persons" | "logs" | "events" | "configs";

type Summary = {
  person_count: number;
  log_count: number;
  pending_event_count: number;
  today_late_return_count: number;
};

type Person = {
  person_id: string;
  name: string;
  role: string;
  department: string;
  status: string;
  valid_from?: string | null;
  valid_until?: string | null;
  consent_status: string;
  updated_at?: string;
};

type RecognitionLog = {
  id: number;
  event_time: string;
  camera_id: string;
  location: string;
  person_id?: string | null;
  person_name?: string | null;
  similarity?: number | null;
  result_type: string;
};

type SecurityEvent = {
  id: number;
  event_type: string;
  event_time: string;
  person_id?: string | null;
  person_name?: string | null;
  camera_id: string;
  location: string;
  confidence?: number | null;
  review_status: string;
  reviewer?: string | null;
  review_comment?: string | null;
};

type FaceEmbeddingMeta = {
  id: number;
  person_id: string;
  image_path: string;
  angle: string;
  quality_score: number;
  model_name: string;
  created_at: string;
};

type EnrollResult = {
  person_id: string;
  image_path: string;
  angle: string;
  quality_score: number;
  model_name: string;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = init?.body instanceof FormData
    ? init?.headers
    : { "Content-Type": "application/json", ...(init?.headers ?? {}) };
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

function App() {
  const [view, setView] = useState<View>("dashboard");

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">Face Access</div>
        <nav>
          <button className={view === "dashboard" ? "active" : ""} onClick={() => setView("dashboard")}>仪表盘</button>
          <button className={view === "persons" ? "active" : ""} onClick={() => setView("persons")}>人员管理</button>
          <button className={view === "logs" ? "active" : ""} onClick={() => setView("logs")}>识别日志</button>
          <button className={view === "events" ? "active" : ""} onClick={() => setView("events")}>安全事件</button>
          <button className={view === "configs" ? "active" : ""} onClick={() => setView("configs")}>系统配置</button>
        </nav>
      </aside>
      <main className="main">
        {view === "dashboard" && <Dashboard />}
        {view === "persons" && <Persons />}
        {view === "logs" && <Logs />}
        {view === "events" && <Events />}
        {view === "configs" && <Configs />}
      </main>
    </div>
  );
}

function Dashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api<Summary>("/stats/summary").then(setSummary).catch((error) => setError(error.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!summary) return <Loading />;

  return (
    <section>
      <Header title="仪表盘" subtitle="系统运行概览" />
      <div className="stats-grid">
        <Stat label="人员总数" value={summary.person_count} tone="green" />
        <Stat label="识别日志" value={summary.log_count} tone="blue" />
        <Stat label="待复核事件" value={summary.pending_event_count} tone="amber" />
        <Stat label="今日晚归疑似" value={summary.today_late_return_count} tone="red" />
      </div>
    </section>
  );
}

function Persons() {
  const emptyForm = useMemo<Person>(() => ({
    person_id: "",
    name: "",
    role: "student",
    department: "",
    status: "active",
    valid_from: "",
    valid_until: "",
    consent_status: "granted"
  }), []);
  const [persons, setPersons] = useState<Person[]>([]);
  const [form, setForm] = useState<Person>(emptyForm);
  const [selectedPersonId, setSelectedPersonId] = useState("");
  const [detailForm, setDetailForm] = useState<Person | null>(null);
  const [embeddings, setEmbeddings] = useState<FaceEmbeddingMeta[]>([]);
  const [angle, setAngle] = useState("front");
  const [file, setFile] = useState<File | null>(null);
  const [enrollResult, setEnrollResult] = useState<EnrollResult | null>(null);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [savingDetail, setSavingDetail] = useState(false);
  const [submittingFace, setSubmittingFace] = useState(false);

  const selectedPerson = persons.find((person) => person.person_id === selectedPersonId) ?? null;

  const load = async (preferredPersonId = selectedPersonId) => {
    try {
      const items = await api<Person[]>("/persons");
      setPersons(items);
      if (preferredPersonId && items.some((person) => person.person_id === preferredPersonId)) {
        setSelectedPersonId(preferredPersonId);
      } else if (items.length > 0) {
        setSelectedPersonId(items[0].person_id);
      } else {
        setSelectedPersonId("");
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (!selectedPerson) {
      setDetailForm(null);
      setEmbeddings([]);
      setDetailError("");
      setEnrollResult(null);
      return;
    }
    setDetailForm({ ...selectedPerson });
    setDetailError("");
    setEnrollResult(null);
    api<FaceEmbeddingMeta[]>(`/persons/${selectedPerson.person_id}/embeddings`)
      .then(setEmbeddings)
      .catch((error) => setDetailError(error instanceof Error ? error.message : String(error)));
  }, [selectedPersonId, selectedPerson]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      const person = await api<Person>("/persons", { method: "POST", body: JSON.stringify(form) });
      setForm(emptyForm);
      await load(person.person_id);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  };

  const saveDetail = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!detailForm) return;
    setSavingDetail(true);
    setDetailError("");
    try {
      const person = await api<Person>(`/persons/${detailForm.person_id}`, {
        method: "PUT",
        body: JSON.stringify(detailForm)
      });
      setDetailForm(person);
      await load(person.person_id);
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : String(error));
    } finally {
      setSavingDetail(false);
    }
  };

  const uploadFace = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedPersonId || !file) return;
    setSubmittingFace(true);
    setDetailError("");
    setEnrollResult(null);

    const formData = new FormData();
    formData.append("angle", angle);
    formData.append("image", file);

    try {
      const uploadResult = await api<EnrollResult>(`/persons/${selectedPersonId}/face-images`, {
        method: "POST",
        body: formData
      });
      setEnrollResult(uploadResult);
      setFile(null);
      const latest = await api<FaceEmbeddingMeta[]>(`/persons/${selectedPersonId}/embeddings`);
      setEmbeddings(latest);
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmittingFace(false);
    }
  };

  return (
    <section>
      <Header title="人员管理" subtitle="维护人员资料、注册采集人脸图片并查看已生成特征" />
      {error && <ErrorState message={error} />}
      <div className="person-workspace">
        <div className="person-list-pane">
          <form className="inline-form person-create-form" onSubmit={submit}>
            <input placeholder="学号/工号" value={form.person_id} onChange={(e) => setForm({ ...form, person_id: e.target.value })} required />
            <input placeholder="姓名" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
            <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
              <option value="student">student</option>
              <option value="teacher">teacher</option>
              <option value="admin">admin</option>
              <option value="visitor">visitor</option>
            </select>
            <input placeholder="院系/部门" value={form.department} onChange={(e) => setForm({ ...form, department: e.target.value })} />
            <select value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value })}>
              <option value="active">active</option>
              <option value="visitor">visitor</option>
              <option value="expired">expired</option>
              <option value="graduated">graduated</option>
              <option value="disabled">disabled</option>
            </select>
            <input placeholder="有效期至" value={form.valid_until ?? ""} onChange={(e) => setForm({ ...form, valid_until: e.target.value })} />
            <button type="submit">新增人员</button>
          </form>
          <DataTable
            columns={["person_id", "name", "role", "department", "status", "valid_until"]}
            rows={persons}
            empty="暂无人员数据"
            selectedKey={selectedPersonId}
            getRowKey={(row, index) => String(row.person_id ?? index)}
            onRowClick={(row) => setSelectedPersonId(String(row.person_id))}
          />
        </div>

        <aside className="person-detail-pane">
          {!detailForm && <div className="state">请选择一名人员查看详情和注册采集</div>}
          {detailForm && (
            <>
              <div className="detail-heading">
                <div>
                  <h2>{detailForm.name || detailForm.person_id}</h2>
                  <p>{detailForm.person_id}</p>
                </div>
                <Status value={detailForm.status} />
              </div>
              {detailError && <ErrorState message={detailError} />}
              {enrollResult && (
                <div className="success">
                  注册成功：{enrollResult.angle}，质量分 {formatNumber(enrollResult.quality_score)}，模型 {enrollResult.model_name}
                </div>
              )}
              <form className="detail-form" onSubmit={saveDetail}>
                <label>
                  学号/工号
                  <input value={detailForm.person_id} disabled />
                </label>
                <label>
                  姓名
                  <input value={detailForm.name} onChange={(event) => setDetailForm({ ...detailForm, name: event.target.value })} required />
                </label>
                <label>
                  角色
                  <select value={detailForm.role} onChange={(event) => setDetailForm({ ...detailForm, role: event.target.value })}>
                    <option value="student">student</option>
                    <option value="teacher">teacher</option>
                    <option value="admin">admin</option>
                    <option value="visitor">visitor</option>
                  </select>
                </label>
                <label>
                  院系/部门
                  <input value={detailForm.department} onChange={(event) => setDetailForm({ ...detailForm, department: event.target.value })} />
                </label>
                <label>
                  状态
                  <select value={detailForm.status} onChange={(event) => setDetailForm({ ...detailForm, status: event.target.value })}>
                    <option value="active">active</option>
                    <option value="visitor">visitor</option>
                    <option value="expired">expired</option>
                    <option value="graduated">graduated</option>
                    <option value="disabled">disabled</option>
                  </select>
                </label>
                <label>
                  有效期至
                  <input value={detailForm.valid_until ?? ""} onChange={(event) => setDetailForm({ ...detailForm, valid_until: event.target.value })} />
                </label>
                <label>
                  授权状态
                  <select value={detailForm.consent_status} onChange={(event) => setDetailForm({ ...detailForm, consent_status: event.target.value })}>
                    <option value="granted">granted</option>
                    <option value="pending">pending</option>
                    <option value="revoked">revoked</option>
                  </select>
                </label>
                <button type="submit" disabled={savingDetail}>{savingDetail ? "保存中..." : "保存资料"}</button>
              </form>

              <form className="enroll-form detail-enroll-form" onSubmit={uploadFace}>
                <label>
                  角度
                  <select value={angle} onChange={(event) => setAngle(event.target.value)}>
                    <option value="front">front</option>
                    <option value="left_45">left_45</option>
                    <option value="right_45">right_45</option>
                    <option value="mask">mask</option>
                  </select>
                </label>
                <label>
                  图片
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                  />
                </label>
                <button type="submit" disabled={!file || submittingFace}>
                  {submittingFace ? "注册中..." : "上传并注册"}
                </button>
              </form>

              <DataTable
                columns={["created_at", "angle", "quality_score", "model_name", "image_path"]}
                rows={embeddings.map((item) => ({ ...item, quality_score: formatNumber(item.quality_score) }))}
                empty="该人员暂无注册特征"
              />
            </>
          )}
        </aside>
      </div>
    </section>
  );
}

function Logs() {
  const [logs, setLogs] = useState<RecognitionLog[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    api<RecognitionLog[]>("/recognition-logs").then(setLogs).catch((error) => setError(error.message));
  }, []);
  return (
    <section>
      <Header title="识别日志" subtitle="最近识别结果" />
      {error && <ErrorState message={error} />}
      <DataTable
        columns={["event_time", "camera_id", "location", "person_name", "similarity", "result_type"]}
        rows={logs.map((log) => ({ ...log, similarity: formatNumber(log.similarity) }))}
        empty="暂无识别日志"
      />
    </section>
  );
}

function Events() {
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [error, setError] = useState("");
  const [comment, setComment] = useState("人工复核");

  const load = () => api<SecurityEvent[]>("/security-events").then(setEvents).catch((error) => setError(error.message));
  useEffect(() => {
    void load();
  }, []);

  const review = async (id: number, reviewStatus: "confirmed" | "rejected") => {
    await api<SecurityEvent>(`/security-events/${id}/review`, {
      method: "PUT",
      body: JSON.stringify({ review_status: reviewStatus, reviewer: "admin", review_comment: comment })
    });
    await load();
  };

  return (
    <section>
      <Header title="安全事件" subtitle="陌生人员和晚归疑似事件复核" />
      {error && <ErrorState message={error} />}
      <div className="review-bar">
        <label>复核备注</label>
        <input value={comment} onChange={(event) => setComment(event.target.value)} />
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>时间</th>
              <th>类型</th>
              <th>人员</th>
              <th>地点</th>
              <th>置信度</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {events.map((event) => (
              <tr key={event.id}>
                <td>{event.event_time}</td>
                <td>{event.event_type}</td>
                <td>{event.person_name || event.person_id || "unknown"}</td>
                <td>{event.location}</td>
                <td>{formatNumber(event.confidence)}</td>
                <td><Status value={event.review_status} /></td>
                <td className="actions">
                  <button onClick={() => review(event.id, "confirmed")} disabled={event.review_status !== "pending"}>确认</button>
                  <button className="secondary" onClick={() => review(event.id, "rejected")} disabled={event.review_status !== "pending"}>驳回</button>
                </td>
              </tr>
            ))}
            {events.length === 0 && <tr><td colSpan={7} className="empty">暂无安全事件</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Configs() {
  const [configs, setConfigs] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api<Record<string, unknown>>("/configs").then(setConfigs).catch((error) => setError(error.message));
  }, []);
  return (
    <section>
      <Header title="系统配置" subtitle="当前运行配置，只读展示" />
      {error && <ErrorState message={error} />}
      <pre className="config-view">{configs ? JSON.stringify(configs, null, 2) : "加载中..."}</pre>
    </section>
  );
}

function Header({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </header>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className={`stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DataTable({
  columns,
  rows,
  empty,
  selectedKey,
  getRowKey,
  onRowClick
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  empty: string;
  selectedKey?: string;
  getRowKey?: (row: Record<string, unknown>, index: number) => string;
  onRowClick?: (row: Record<string, unknown>) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const rowKey = getRowKey ? getRowKey(row, index) : String(row.id ?? row.person_id ?? index);
            return (
              <tr
                key={rowKey}
                className={`${onRowClick ? "clickable-row" : ""} ${selectedKey === rowKey ? "selected-row" : ""}`}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}
              </tr>
            );
          })}
          {rows.length === 0 && <tr><td colSpan={columns.length} className="empty">{empty}</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

function Status({ value }: { value: string }) {
  return <span className={`status ${value}`}>{value}</span>;
}

function Loading() {
  return <div className="state">加载中...</div>;
}

function ErrorState({ message }: { message: string }) {
  return <div className="error">{message}</div>;
}

function formatNumber(value?: number | null) {
  return typeof value === "number" ? value.toFixed(2) : "";
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
