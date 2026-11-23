const base = process.env.E2E_BASE_URL;
const password = process.env.E2E_ADMIN_PASSWORD;
if (!base || !password) throw new Error("E2E_BASE_URL and E2E_ADMIN_PASSWORD are required");

const headers = { "content-type": "application/json", origin: new URL(base).origin };
const cookies = new Map();
const remember = (response) => {
  for (const line of response.headers.getSetCookie()) {
    const [pair] = line.split(";");
    const index = pair.indexOf("=");
    cookies.set(pair.slice(0, index), pair.slice(index + 1));
  }
};
const cookieHeader = () => [...cookies].map(([key, value]) => `${key}=${value}`).join("; ");
const call = async (method, path, body, extra = {}) => {
  const response = await fetch(`${base}/api${path}`, { method, headers: { ...headers, cookie: cookieHeader(), ...extra }, body: body ? JSON.stringify(body) : undefined });
  remember(response);
  return response;
};

let login = await call("POST", "/auth/login", { identifier: "admin", password });
if (login.status === 401) login = await call("POST", "/auth/login", { identifier: "admin", password: "admin" });
if (!login.ok) throw new Error(`admin login failed: ${login.status}`);
const profile = await login.json();
if (profile.requires_password_change) {
  const changed = await call("PUT", "/auth/password", { current_password: "admin", new_password: password }, { "X-CSRF-Token": cookies.get("abylkhaiyr_csrf") ?? "" });
  if (!changed.ok) throw new Error(`password change failed: ${changed.status}`);
}
const username = process.env.E2E_USER ?? "e2e_pages_user";
const registered = await call("POST", "/auth/register", { firstName: "Page", lastName: "Audit", email: `${username}@example.com`, username, password: "e2e-user-password-1" });
if (![201, 409, 429].includes(registered.status)) throw new Error(`registration failed: ${registered.status}`);
console.log("e2e bootstrap complete");
