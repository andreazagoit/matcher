"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EyeIcon, EyeOffIcon, CopyIcon, RefreshCwIcon, UsersIcon, PlayIcon, CheckCircleIcon, XCircleIcon, Loader2Icon, ExternalLinkIcon } from "lucide-react";
import { CodeBlock } from "@/components/ui/code-block";

interface App {
  id: string;
  name: string;
  description?: string;
  clientId: string;
  secretKey: string;
  redirectUris: string[];
  accessTokenTtl: string;
  refreshTokenTtl: string;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

interface AppStats {
  activeAccessTokens: number;
  activeRefreshTokens: number;
  totalTokensIssued: number;
  authorizedUsersCount: number;
}

interface AuthorizedUser {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  authorizedAt: string;
  lastActivity?: string;
}

type Tab = "overview" | "settings" | "users";

export default function AppDetailPage() {
  const params = useParams();
  const router = useRouter();
  const appId = params.appId as string;

  const [app, setApp] = useState<App | null>(null);
  const [stats, setStats] = useState<AppStats | null>(null);
  const [users, setUsers] = useState<AuthorizedUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [showSecret, setShowSecret] = useState(false);
  const [testingApi, setTestingApi] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string; data?: Array<{ id: string; firstName: string; lastName: string; email: string }> } | null>(null);

  useEffect(() => {
    async function fetchApp() {
      try {
        const res = await fetch(`/api/dashboard/apps/${appId}`);
        if (res.ok) {
          const data = await res.json();
          setApp(data.app);
          setStats(data.stats);
        } else {
          router.push("/dashboard");
        }
      } catch (error) {
        console.error("Failed to fetch app:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchApp();
  }, [appId, router]);

  // Fetch users when users tab is selected
  useEffect(() => {
    if (activeTab === "users" && users.length === 0 && !loadingUsers) {
      fetchUsers();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  const fetchUsers = async () => {
    setLoadingUsers(true);
    try {
      const res = await fetch(`/api/dashboard/apps/${appId}/users`);
      if (res.ok) {
        const data = await res.json();
        setUsers(data.users);
      }
    } catch (error) {
      console.error("Failed to fetch users:", error);
    } finally {
      setLoadingUsers(false);
    }
  };

  const handleRotateSecret = async () => {
    if (!confirm("Are you sure? This will invalidate the current secret key. All integrations will need to be updated.")) return;

    try {
      const res = await fetch(`/api/dashboard/apps/${appId}/rotate-secret`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setApp((prev) => prev ? { ...prev, secretKey: data.secretKey } : null);
        setShowSecret(true);
      }
    } catch (error) {
      console.error("Failed to rotate secret:", error);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete this app? This cannot be undone.")) return;

    try {
      const res = await fetch(`/api/dashboard/apps/${appId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        router.push("/dashboard");
      }
    } catch (error) {
      console.error("Failed to delete app:", error);
    }
  };

  const handleRevokeUserAccess = async (userId: string) => {
    if (!confirm("Revoke this user's access? They will need to re-authorize.")) return;

    try {
      const res = await fetch(`/api/dashboard/apps/${appId}/users/${userId}/revoke`, {
        method: "POST",
      });
      if (res.ok) {
        setUsers((prev) => prev.filter((u) => u.id !== userId));
        if (stats) {
          setStats({ ...stats, authorizedUsersCount: stats.authorizedUsersCount - 1 });
        }
      }
    } catch (error) {
      console.error("Failed to revoke user access:", error);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  // Generate PKCE code verifier and challenge
  const generatePKCE = async () => {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    const codeVerifier = btoa(String.fromCharCode(...array))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');

    const encoder = new TextEncoder();
    const data = encoder.encode(codeVerifier);
    const digest = await crypto.subtle.digest('SHA-256', data);
    const codeChallenge = btoa(String.fromCharCode(...new Uint8Array(digest)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=/g, '');

    return { codeVerifier, codeChallenge };
  };

  const testOAuthFlow = async () => {
    if (!app) return;

    const testCallbackUri = `${window.location.origin}/dashboard/oauth-test-callback`;

    // Check if test callback URI is in redirect URIs
    if (!app.redirectUris?.includes(testCallbackUri)) {
      // Add it automatically
      try {
        const updatedUris = [...(app.redirectUris || []), testCallbackUri];
        const res = await fetch(`/api/dashboard/apps/${appId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ redirectUris: updatedUris }),
        });
        if (res.ok) {
          setApp(prev => prev ? { ...prev, redirectUris: updatedUris } : null);
        }
      } catch (error) {
        console.error('Failed to add test callback URI:', error);
      }
    }

    const { codeVerifier, codeChallenge } = await generatePKCE();
    const state = crypto.randomUUID();

    // Store for callback handling
    sessionStorage.setItem('oauth_test_state', state);
    sessionStorage.setItem('oauth_test_code_verifier', codeVerifier);
    sessionStorage.setItem('oauth_test_client_id', app.clientId);

    // Build OAuth URL
    const authUrl = new URL(`${window.location.origin}/oauth/authorize`);
    authUrl.searchParams.set('client_id', app.clientId);
    authUrl.searchParams.set('redirect_uri', testCallbackUri);
    authUrl.searchParams.set('response_type', 'code');
    authUrl.searchParams.set('scope', 'openid profile read:matches');
    authUrl.searchParams.set('state', state);
    authUrl.searchParams.set('code_challenge', codeChallenge);
    authUrl.searchParams.set('code_challenge_method', 'S256');

    // Open in new popup window
    const width = 500;
    const height = 700;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;
    window.open(
      authUrl.toString(),
      'oauth_test',
      `width=${width},height=${height},left=${left},top=${top},popup=yes`
    );
  };

  const testApiConnection = async () => {
    if (!app) return;

    setTestingApi(true);
    setTestResult(null);

    try {
      const response = await fetch("/api/graphql", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${app.secretKey}`,
        },
        body: JSON.stringify({
          query: `query { users { id firstName lastName email } }`,
        }),
      });

      const result = await response.json();

      if (result.errors) {
        setTestResult({
          success: false,
          message: result.errors[0]?.message || "GraphQL error",
        });
      } else if (result.data?.users) {
        setTestResult({
          success: true,
          message: `Connection successful! Found ${result.data.users.length} users.`,
          data: result.data.users.slice(0, 3), // Show first 3 users
        });
      } else {
        setTestResult({
          success: false,
          message: "Unexpected response format",
        });
      }
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : "Connection failed",
      });
    } finally {
      setTestingApi(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-8 bg-muted rounded w-1/3 mb-8"></div>
        <div className="h-64 bg-card rounded"></div>
      </div>
    );
  }

  if (!app) return null;

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "settings", label: "Settings" },
    { id: "users", label: "Users" },
  ];

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground mb-6">
        <Link href="/dashboard" className="hover:text-foreground transition">Apps</Link>
        <span>/</span>
        <span className="text-foreground">{app.name}</span>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-xl bg-primary flex items-center justify-center text-primary-foreground text-2xl font-bold">
            {app.name.charAt(0).toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground flex items-center gap-3">
              {app.name}
              <Badge variant={app.isActive ? "default" : "secondary"}>
                {app.isActive ? "Active" : "Inactive"}
              </Badge>
            </h1>
            {app.description && (
              <p className="text-muted-foreground mt-1">{app.description}</p>
            )}
          </div>
        </div>
        <Button variant="destructive" onClick={handleDelete}>
          Delete App
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 text-sm font-medium transition-colors relative ${activeTab === tab.id
              ? "text-primary"
              : "text-muted-foreground hover:text-foreground"
              }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary"></div>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "overview" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stats */}
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Active Tokens</CardDescription>
              <CardTitle className="text-3xl">{stats?.activeAccessTokens || 0}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Authorized Users</CardDescription>
              <CardTitle className="text-3xl">{stats?.authorizedUsersCount || 0}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Tokens Issued</CardDescription>
              <CardTitle className="text-3xl">{stats?.totalTokensIssued || 0}</CardTitle>
            </CardHeader>
          </Card>

          {/* Credentials */}
          <Card className="lg:col-span-3">
            <CardHeader>
              <CardTitle>Credentials</CardTitle>
              <CardDescription>Use these to integrate with your app</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Client ID */}
              <div className="space-y-2">
                <Label>Client ID</Label>
                <p className="text-xs text-muted-foreground">For OAuth login flow. Can be exposed in frontend.</p>
                <InputGroup>
                  <InputGroupInput
                    value={app.clientId}
                    readOnly
                    className="font-mono"
                  />
                  <InputGroupAddon align="inline-end">
                    <InputGroupButton
                      size="icon-xs"
                      variant="ghost"
                      onClick={() => copyToClipboard(app.clientId)}
                      aria-label="Copy"
                    >
                      <CopyIcon className="h-4 w-4" />
                    </InputGroupButton>
                  </InputGroupAddon>
                </InputGroup>
              </div>

              <Separator />

              {/* Secret Key */}
              <div className="space-y-2">
                <Label>Secret Key</Label>
                <p className="text-xs text-muted-foreground">For direct API access. Keep secret, backend only!</p>
                <InputGroup>
                  <InputGroupInput
                    value={showSecret ? app.secretKey : "•".repeat(app.secretKey.length)}
                    readOnly
                    type={showSecret ? "text" : "password"}
                    className="font-mono"
                  />
                  <InputGroupAddon align="inline-end">
                    <InputGroupButton
                      size="icon-xs"
                      variant="ghost"
                      onClick={() => setShowSecret(!showSecret)}
                      aria-label={showSecret ? "Hide" : "Show"}
                    >
                      {showSecret ? (
                        <EyeOffIcon className="h-4 w-4" />
                      ) : (
                        <EyeIcon className="h-4 w-4" />
                      )}
                    </InputGroupButton>
                    <InputGroupButton
                      size="icon-xs"
                      variant="ghost"
                      onClick={() => copyToClipboard(app.secretKey)}
                      aria-label="Copy"
                    >
                      <CopyIcon className="h-4 w-4" />
                    </InputGroupButton>
                    <InputGroupButton
                      size="icon-xs"
                      variant="ghost"
                      onClick={handleRotateSecret}
                      aria-label="Rotate"
                    >
                      <RefreshCwIcon className="h-4 w-4" />
                    </InputGroupButton>
                  </InputGroupAddon>
                </InputGroup>
              </div>
            </CardContent>
          </Card>

          {/* Quick Start - OAuth Flow */}
          <Card className="lg:col-span-3">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>OAuth Login</CardTitle>
                  <CardDescription>Autentica gli utenti con il loro account Matcher</CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={testOAuthFlow}>
                  <ExternalLinkIcon className="h-4 w-4 mr-2" />
                  Test
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <CodeBlock
                language="tsx"
                code={`import { useCallback } from "react";

const CLIENT_ID = "${app.clientId}";
const REDIRECT_URI = "https://yourapp.com/callback";
const AUTH_URL = "${process.env.NEXT_PUBLIC_APP_URL || window.location.origin}/oauth/authorize";
const TOKEN_URL = "${process.env.NEXT_PUBLIC_APP_URL || window.location.origin}/oauth/token";

// Login button component
function LoginButton() {
  const handleLogin = useCallback(async () => {
    // Generate PKCE
    const verifier = crypto.randomUUID() + crypto.randomUUID();
    const challenge = btoa(String.fromCharCode(
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)))
    )).replace(/[+/=]/g, c => ({ "+": "-", "/": "_", "=": "" }[c]!));
    
    sessionStorage.setItem("pkce_verifier", verifier);
    
    const params = new URLSearchParams({
      response_type: "code",
      client_id: CLIENT_ID,
      redirect_uri: REDIRECT_URI,
      scope: "openid profile read:matches",
      code_challenge: challenge,
      code_challenge_method: "S256",
    });
    
    window.location.href = \`\${AUTH_URL}?\${params}\`;
  }, []);

  return <button onClick={handleLogin}>Login with Matcher</button>;
}

// Callback page component
async function handleCallback(code: string) {
  const verifier = sessionStorage.getItem("pkce_verifier");
  
  const res = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      client_id: CLIENT_ID,
      code,
      redirect_uri: REDIRECT_URI,
      code_verifier: verifier!,
    }),
  });
  
  const { access_token } = await res.json();
  localStorage.setItem("token", access_token);
}`}
              />
            </CardContent>
          </Card>

          {/* Quick Start - M2M */}
          <Card className="lg:col-span-3">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Server API</CardTitle>
                  <CardDescription>Accesso diretto con Secret Key (backend only)</CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={testApiConnection} disabled={testingApi}>
                  {testingApi ? <Loader2Icon className="h-4 w-4 animate-spin" /> : <PlayIcon className="h-4 w-4" />}
                  <span className="ml-2">{testingApi ? "Testing..." : "Test"}</span>
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {testResult && (
                <div className={`p-3 rounded-lg text-sm ${testResult.success ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"}`}>
                  {testResult.success ? <CheckCircleIcon className="h-4 w-4 inline mr-2" /> : <XCircleIcon className="h-4 w-4 inline mr-2" />}
                  {testResult.message}
                </div>
              )}
              <CodeBlock
                language="typescript"
                code={`// server.ts - Node.js / Next.js API Route
const API_URL = "${process.env.NEXT_PUBLIC_APP_URL || window.location.origin}/api/graphql";
const SECRET_KEY = process.env.MATCHER_SECRET_KEY; // "${showSecret ? app.secretKey : "sk_..."}";

async function getUsers() {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": \`Bearer \${SECRET_KEY}\`,
    },
    body: JSON.stringify({
      query: \`{ users { id firstName lastName email } }\`,
    }),
  });
  
  const { data } = await res.json();
  return data.users;
}

async function findMatches(userId: string) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": \`Bearer \${SECRET_KEY}\`,
    },
    body: JSON.stringify({
      query: \`query($id: ID!) { findMatches(userId: $id) { id firstName } }\`,
      variables: { id: userId },
    }),
  });
  
  const { data } = await res.json();
  return data.findMatches;
}`}
              />
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "settings" && (
        <Card>
          <CardHeader>
            <CardTitle>App Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <Label className="mb-2 block">Redirect URIs</Label>
              <div className="space-y-2">
                {app.redirectUris?.length ? (
                  app.redirectUris.map((uri, i) => (
                    <code key={i} className="block bg-muted p-3 rounded text-primary font-mono text-sm">
                      {uri}
                    </code>
                  ))
                ) : (
                  <p className="text-muted-foreground">No redirect URIs configured</p>
                )}
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Access Token TTL</p>
                <p className="text-foreground font-medium">{parseInt(app.accessTokenTtl) / 60} minutes</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-1">Refresh Token TTL</p>
                <p className="text-foreground font-medium">{parseInt(app.refreshTokenTtl) / 86400} days</p>
              </div>
            </div>

            <Separator />

            <Link href={`/dashboard/${appId}/edit`}>
              <Button>Edit Settings</Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {activeTab === "users" && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <UsersIcon className="h-5 w-5" />
                  Authorized Users
                </CardTitle>
                <CardDescription>Users who have granted access to this app</CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={fetchUsers}>
                <RefreshCwIcon className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {loadingUsers ? (
              <div className="py-8 text-center text-muted-foreground">Loading users...</div>
            ) : users.length === 0 ? (
              <div className="py-8 text-center text-muted-foreground">
                <UsersIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No users have authorized this app yet.</p>
                <p className="text-sm mt-2">Users will appear here after they complete the OAuth flow.</p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>User</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>Authorized</TableHead>
                    <TableHead>Last Activity</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {users.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell className="font-medium">
                        {user.firstName} {user.lastName}
                      </TableCell>
                      <TableCell className="text-muted-foreground">{user.email}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {new Date(user.authorizedAt).toLocaleDateString()}
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {user.lastActivity
                          ? new Date(user.lastActivity).toLocaleDateString()
                          : "-"
                        }
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          onClick={() => handleRevokeUserAccess(user.id)}
                        >
                          Revoke
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
