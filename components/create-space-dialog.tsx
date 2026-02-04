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
import { Switch } from "@/components/ui/switch";
import { graphql } from "@/lib/graphql/client";

interface CreateSpaceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: () => void;
}

export function CreateSpaceDialog({ open, onOpenChange, onCreated }: CreateSpaceDialogProps) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    name: "",
    slug: "",
    description: "",
    isPublic: true,
  });

  const resetForm = () => {
    setFormData({
      name: "",
      slug: "",
      description: "",
      isPublic: true,
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
      const data = await graphql<{ createSpace: { id: string } }>(`
        mutation CreateSpace($input: CreateSpaceInput!) {
          createSpace(input: $input) {
            id
            name
            slug
          }
        }
      `, {
        input: {
          name: formData.name,
          slug: formData.slug || undefined,
          description: formData.description,
          isPublic: formData.isPublic,
        }
      });

      onCreated?.();
      handleOpenChange(false);

      // Navigate directly to the space page
      router.push(`/dashboard/${data.createSpace.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create space");
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Create New Space</DialogTitle>
          <DialogDescription>
            Create a new community, club, or organization.
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
              <Label htmlFor="name">Space Name *</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
                placeholder="FitLife Club"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="slug">URL Slug (optional)</Label>
              <Input
                id="slug"
                value={formData.slug}
                onChange={(e) => setFormData({ ...formData, slug: e.target.value })}
                placeholder="fitlife-club"
              />
              <p className="text-xs text-muted-foreground">Will be auto-generated if left empty</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="A community for fitness enthusiasts"
              />
            </div>

            <div className="flex items-center space-x-2 pt-2">
              <Switch
                id="is-public"
                checked={formData.isPublic}
                onCheckedChange={(checked) => setFormData({ ...formData, isPublic: checked })}
              />
              <Label htmlFor="is-public">Public Space</Label>
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
              {loading ? "Creating..." : "Create Space"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
