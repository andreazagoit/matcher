"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface CreateAppDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: () => void;
}

export function CreateAppDialog({ open, onOpenChange, onCreated }: CreateAppDialogProps) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    name: "",
    description: "",
    redirectUris: "",
  });

  const resetForm = () => {
    setFormData({
      name: "",
      description: "",
      redirectUris: "",
    });
    setError(null);
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
      resetForm();
    }
    onOpenChange(newOpen);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/dashboard/clients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          redirectUris: formData.redirectUris
            .split("\n")
            .map((uri) => uri.trim())
            .filter(Boolean),
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to create app");
      }

      const data = await res.json();
      
      // Store secret key temporarily to show on the app page
      sessionStorage.setItem(`new_secret_${data.app.id}`, data.credentials.secretKey);
      
      onCreated?.();
      handleOpenChange(false);
      
      // Navigate directly to the app page
      router.push(`/dashboard/${data.app.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create app");
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Create New App</DialogTitle>
          <DialogDescription>
            You'll get credentials for both OAuth and M2M access.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-5 mt-4">
          {error && (
            <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded text-sm">
              {error}
            </div>
          )}

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">App Name *</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
                placeholder="My Awesome App"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="What does your app do?"
              />
            </div>

            <div className="space-y-2">
              <Label>Redirect URIs</Label>
              <Textarea
                value={formData.redirectUris}
                onChange={(e) => setFormData({ ...formData, redirectUris: e.target.value })}
                className="font-mono text-sm h-20"
                placeholder={"https://myapp.com/callback\nhttp://localhost:3000/callback"}
              />
              <p className="text-xs text-muted-foreground">One URI per line. Required for OAuth login flow.</p>
            </div>
          </div>

          <div className="flex gap-3 pt-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={loading || !formData.name}
              className="flex-1"
            >
              {loading ? "Creating..." : "Create App"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
