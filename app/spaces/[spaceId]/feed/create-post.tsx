import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { graphql } from "@/lib/graphql/client"
import { useSession } from "next-auth/react"
import { Loader2Icon, SendIcon } from "lucide-react"
import { Card } from "@/components/ui/card"

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
                mutation CreatePost($spaceId: ID!, $content: String!) {
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
        <Card className="mb-6 border-none shadow-sm overflow-hidden py-0">
            <div className="p-4 flex gap-4 items-center">
                <Avatar className="shrink-0">
                    <AvatarFallback className="bg-primary/10 text-primary">
                        {session.user.name?.[0] || "U"}
                    </AvatarFallback>
                </Avatar>

                <div className="flex-1 relative">
                    <form
                        onSubmit={(e) => {
                            e.preventDefault()
                            handleSubmit()
                        }}
                        className="flex items-center gap-2"
                    >
                        <input
                            type="text"
                            placeholder="Start a post..."
                            value={content}
                            onChange={(e) => setContent(e.target.value)}
                            disabled={isSubmitting}
                            className="flex-1 h-11 rounded-full bg-muted/30 hover:bg-muted/50 focus:bg-background border-none px-5 text-sm transition-all focus:ring-2 focus:ring-primary/10 focus:outline-none placeholder:text-muted-foreground/70"
                        />
                        {content.trim() && (
                            <Button
                                size="icon"
                                type="submit"
                                disabled={isSubmitting}
                                className="h-10 w-10 shrink-0 rounded-full animate-in fade-in zoom-in duration-200"
                            >
                                {isSubmitting ? (
                                    <Loader2Icon className="h-4 w-4 animate-spin" />
                                ) : (
                                    <SendIcon className="h-4 w-4" />
                                )}
                            </Button>
                        )}
                    </form>
                </div>
            </div>
        </Card>
    )
}
