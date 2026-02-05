"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { graphql } from "@/lib/graphql/client"
import { useSession } from "next-auth/react"
import { Loader2Icon, SendIcon } from "lucide-react"

interface CreatePostProps {
    spaceId: string
    onPostCreated: () => void
}

export function CreatePost({ spaceId, onPostCreated }: CreatePostProps) {
    const { data: session } = useSession()
    const [content, setContent] = useState("")
    const [isSubmitting, setIsSubmitting] = useState(false)

    const handleSubmit = async () => {
        if (!content.trim()) return

        setIsSubmitting(true)
        try {
            await graphql(`
                mutation CreatePost($spaceId: String!, $content: String!) {
                    createPost(spaceId: $spaceId, content: $content) {
                        id
                    }
                }
            `, {
                spaceId,
                content: content.trim(),
            })

            setContent("")
            onPostCreated()
        } catch (error) {
            console.error("Failed to create post:", error)
            alert("Failed to create post")
        } finally {
            setIsSubmitting(false)
        }
    }

    if (!session?.user) return null

    return (
        <Card className="mb-6">
            <CardContent className="pt-6">
                <div className="flex gap-4">
                    <Avatar>
                        <AvatarFallback>
                            {session.user.name?.[0] || "U"}
                        </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 space-y-2">
                        <Textarea
                            placeholder="Share something with the community..."
                            value={content}
                            onChange={(e) => setContent(e.target.value)}
                            className="min-h-[100px] resize-none"
                            disabled={isSubmitting}
                        />
                        <div className="flex justify-end">
                            <Button
                                onClick={handleSubmit}
                                disabled={!content.trim() || isSubmitting}
                            >
                                {isSubmitting ? (
                                    <>
                                        <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                                        Posting...
                                    </>
                                ) : (
                                    <>
                                        <SendIcon className="mr-2 h-4 w-4" />
                                        Post
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
