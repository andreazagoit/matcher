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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMutation } from "@apollo/client/react";
import { CREATE_SPACE } from "@/lib/models/spaces/gql";
import type { CreateSpaceMutation, CreateSpaceMutationVariables } from "@/lib/graphql/__generated__/graphql";

interface CreateSpaceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: () => void;
}

export function CreateSpaceDialog({ open, onOpenChange, onCreated }: CreateSpaceDialogProps) {
  const router = useRouter();
  const [createSpace, { loading }] = useMutation<CreateSpaceMutation, CreateSpaceMutationVariables>(CREATE_SPACE);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    name: "",
    slug: "",
    description: "",
    visibility: "public",
    joinPolicy: "open",
  });

  const resetForm = () => {
    setFormData({
      name: "",
      slug: "",
      description: "",
      visibility: "public",
      joinPolicy: "open",
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
    setError(null);

    try {
      const { data } = await createSpace({
        variables: {
          input: {
            name: formData.name,
            slug: formData.slug || undefined,
            description: formData.description,
            visibility: formData.visibility,
            joinPolicy: formData.joinPolicy,
          }
        }
      });

      if (data?.createSpace) {
        onCreated?.();
        handleOpenChange(false);
        // Navigate directly to the space page
        router.push(`/spaces/${data.createSpace.slug}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create space");
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

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="space-y-2">
                <Label htmlFor="visibility">Visibility</Label>
                <Select
                  value={formData.visibility}
                  onValueChange={(value) => setFormData({ ...formData, visibility: value })}
                >
                  <SelectTrigger id="visibility">
                    <SelectValue placeholder="Select visibility" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="public">Public</SelectItem>
                    <SelectItem value="private">Private</SelectItem>
                    <SelectItem value="hidden">Hidden</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="joinPolicy">Join Policy</Label>
                <Select
                  value={formData.joinPolicy}
                  onValueChange={(value) => setFormData({ ...formData, joinPolicy: value })}
                >
                  <SelectTrigger id="joinPolicy">
                    <SelectValue placeholder="Select policy" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="open">Open</SelectItem>
                    <SelectItem value="apply">Apply</SelectItem>
                    <SelectItem value="invite_only">Invite Only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
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
